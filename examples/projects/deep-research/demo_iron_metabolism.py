#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Full gist-shaped deep research demo on locus.

Mirrors the gist at https://gist.github.com/fede-kamel/15ab302e6b4d155f192555a6a6e33cd0
("Memory-Aware Deep Research Agent with ADB + 65K Output") but built
entirely on locus primitives — zero langchain or deepagents imports.

Pipeline:

    1. Ingest ~50 hand-curated iron-metabolism sentences into a fresh
       table in the personal `deepresearch` ADB using
       OCIEmbeddings(model_id="cohere.embed-v4.0").
    2. Wire OracleVectorStore + RAGRetriever, hand them to
       create_deepagent via the new `datastores=` kwarg.
    3. Run Gemini 2.5 Pro with `max_output_tokens=65_536` and ask for a
       long-form research memo with citations to retrieved doc ids.
    4. Save the memo + the tool-call trace to ``output/`` next to this
       script so the artifact survives the run.
    5. Drop the seed table on the way out.

Run:

    export OCI_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=security_token
    export OCI_COMPARTMENT=ocid1.compartment.oc1..xxx
    export OCI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
    export ADB_DSN=<your-adb-tns>
    export ADB_PASSWORD=...
    export ADB_WALLET_LOCATION=~/.oci/wallets/<your-adb>
    uv run python examples/projects/deep-research/demo_iron_metabolism.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import oracledb

from locus.deepagent.factory import create_deepagent
from locus.models import get_model
from locus.rag import OCIEmbeddings, RAGRetriever
from locus.rag.stores.oracle import OracleVectorStore


# A hand-curated medical-knowledge corpus on iron metabolism. ~250 sentences,
# each a single-sentence claim with a clear semantic anchor. Together they
# cover absorption, transport, storage, regulation (hepcidin/ferroportin),
# disorders (deficiency, hemochromatosis, ACD, IRIDA, ferroportin disease,
# transfusional siderosis), diagnostics (lab markers, imaging, genetics),
# treatment modalities (oral iron formulations, IV iron, chelators,
# phlebotomy, ESAs, novel agents), and special populations (pregnancy,
# pediatric, CKD, athletic, geriatric). Sized to give the agent enough
# distinct ground truth to write a 50K+ char memo when asked.
CORPUS = [
    # --- Absorption ---
    "Iron is primarily absorbed in the duodenum and proximal jejunum.",
    "Heme iron from animal sources is absorbed at 15-35%, while non-heme iron from plants is absorbed at only 2-20%.",
    "Vitamin C (ascorbic acid) enhances non-heme iron absorption by reducing ferric (Fe3+) to ferrous (Fe2+) iron.",
    "Phytates in legumes, tannins in tea, and calcium in dairy products all inhibit non-heme iron absorption.",
    "DMT1 (divalent metal transporter 1) is the main apical transporter for non-heme iron in enterocytes.",
    "Heme iron is absorbed via the heme carrier protein HCP1, then broken down intracellularly by heme oxygenase to release iron.",
    "Ferroportin is the only known cellular iron exporter, located on the basolateral membrane of enterocytes and macrophages.",
    # --- Transport ---
    "Transferrin is the main iron transport protein in plasma, binding two ferric ions per molecule.",
    "Transferrin saturation below 16% suggests iron deficiency; saturation above 45% raises concern for iron overload.",
    "Iron is delivered to cells via transferrin receptor 1 (TfR1), which is upregulated in iron-deficient and rapidly dividing cells.",
    "Soluble transferrin receptor (sTfR) is elevated in iron deficiency but unaffected by inflammation, helping distinguish IDA from ACD.",
    # --- Storage ---
    "Ferritin is the primary iron storage protein, sequestering up to 4500 iron atoms per molecule.",
    "Serum ferritin reflects total body iron stores but is an acute-phase reactant elevated by inflammation, infection, and malignancy.",
    "Hemosiderin is an insoluble form of iron storage found in macrophages, accumulating in iron-overload states.",
    "The average adult body contains 3-4 grams of iron; about 65% is in hemoglobin, 25% in storage, 10% in myoglobin and enzymes.",
    # --- Regulation ---
    "Hepcidin, produced by the liver, is the master regulator of iron homeostasis.",
    "Hepcidin binds ferroportin and induces its internalization and degradation, blocking iron export from enterocytes and macrophages.",
    "Hepcidin is upregulated by iron repletion (via BMP6/SMAD signaling) and by inflammation (via IL-6/STAT3 signaling).",
    "Hepcidin is suppressed by iron deficiency, hypoxia, and increased erythropoietic demand (via erythroferrone from erythroblasts).",
    "Erythroferrone, secreted by EPO-stimulated erythroblasts, suppresses hepcidin to mobilize iron for erythropoiesis.",
    "Matriptase-2 (TMPRSS6) cleaves hemojuvelin to suppress hepcidin transcription; loss-of-function mutations cause IRIDA.",
    # --- Iron deficiency / IDA ---
    "Iron deficiency anemia is the most common nutritional deficiency worldwide, affecting an estimated 1.2 billion people.",
    "Classic lab findings in IDA include low ferritin (<30 ng/mL), low transferrin saturation, high TIBC, and microcytic hypochromic RBCs.",
    "Pica (ice, dirt, starch craving) and restless legs syndrome are non-hematologic clinical clues to iron deficiency.",
    "First-line treatment for IDA is oral ferrous sulfate 325 mg three times daily, ideally on an empty stomach with vitamin C.",
    "Alternate-day oral iron dosing improves fractional absorption by avoiding hepcidin elevation triggered by daily doses.",
    "IV iron (ferric carboxymaltose, iron sucrose) is indicated for malabsorption, intolerance, chronic blood loss, or rapid replacement before surgery.",
    # --- Hemochromatosis ---
    "Hereditary hemochromatosis is most commonly caused by C282Y homozygosity in the HFE gene.",
    "HFE-hemochromatosis impairs hepcidin sensing of body iron stores, leading to inappropriately low hepcidin and excessive intestinal iron absorption.",
    "Iron overload in hemochromatosis damages the liver (cirrhosis, HCC), heart (cardiomyopathy, arrhythmias), pancreas (diabetes), joints, and skin.",
    "Diagnosis of HH typically involves transferrin saturation >45%, elevated ferritin, and HFE genetic testing.",
    "Phlebotomy is the first-line treatment for hereditary hemochromatosis, removing roughly 200-250 mg of iron per unit of blood.",
    "Therapeutic phlebotomy target in HH is ferritin below 50 ng/mL with transferrin saturation below 50%.",
    "Iron chelators (deferoxamine, deferasirox, deferiprone) are used when phlebotomy is contraindicated (anemia, cardiac issues) and in transfusion iron overload.",
    # --- Anemia of chronic disease ---
    "Anemia of chronic disease (anemia of inflammation) is caused by elevated hepcidin in chronic inflammatory states, sequestering iron in macrophages.",
    "ACD typically presents as normocytic normochromic anemia with elevated ferritin, low serum iron, low TIBC, and normal or low transferrin saturation.",
    "Treating the underlying inflammation is the cornerstone of ACD management; iron supplementation alone is largely ineffective due to hepcidin block.",
    # --- IRIDA ---
    "Iron-refractory iron deficiency anemia (IRIDA) is an autosomal recessive disorder caused by TMPRSS6 mutations.",
    "In IRIDA, defective matriptase-2 leaves hepcidin inappropriately elevated, so oral iron is poorly absorbed; IV iron is the mainstay of treatment.",
    # --- Pregnancy + special populations ---
    "Iron requirements increase from 18 mg/day in non-pregnant women to 27 mg/day during pregnancy due to fetal demand and plasma volume expansion.",
    "Maternal iron deficiency in pregnancy is associated with preterm birth, low birth weight, and impaired infant neurodevelopment.",
    "Vegetarian and vegan diets typically require 1.8x the iron RDA due to lower bioavailability of non-heme iron.",
    # --- Diagnostics ---
    "MRI T2* relaxometry is the gold standard for non-invasive quantification of liver and cardiac iron load in iron-overload disorders.",
    "Bone marrow iron staining with Prussian blue remains the historical gold standard for assessing iron stores but is rarely needed clinically.",
    "Reticulocyte hemoglobin content (CHr) drops within days of iron deficiency, providing an early functional marker before MCV changes.",
    # --- Extended absorption / transporter biology ---
    "DCYTB (duodenal cytochrome b) is the apical ferric reductase that reduces Fe3+ to Fe2+ before DMT1-mediated uptake.",
    "Hephaestin is a basolateral ceruloplasmin-homolog ferroxidase that oxidizes Fe2+ to Fe3+ for loading onto transferrin after ferroportin export.",
    "Mucin proteins in the duodenal lumen bind ferric iron and keep it soluble, indirectly supporting non-heme absorption.",
    "Ferric citrate is an oral iron formulation that also lowers serum phosphate, used in CKD patients with hyperphosphatemia.",
    "Carbonyl iron is an elemental iron form with lower toxicity profile, sometimes used when ionic iron salts are not tolerated.",
    "Iron polysaccharide complexes (e.g., ferric maltol) offer better GI tolerability with similar efficacy to ferrous sulfate in IBD.",
    "Sucrosomial iron is a novel oral iron formulation in which ferric pyrophosphate is shielded by a phospholipid + sucrester matrix, bypassing DMT1.",
    "Hepcidin levels surge for 24-48 hours after a single oral iron dose, transiently blocking subsequent absorption — the rationale for alternate-day dosing.",
    "Proton-pump inhibitors reduce gastric acid, impairing the reduction of dietary ferric iron and thus non-heme iron absorption.",
    "Bariatric surgery, especially Roux-en-Y gastric bypass, often causes iron deficiency by bypassing the duodenum and reducing acid exposure.",
    "Helicobacter pylori infection can cause refractory iron deficiency anemia even without overt GI bleeding, likely via mucosal sequestration and altered hepcidin.",
    "Celiac disease should be screened for in adult iron deficiency without obvious blood loss, particularly when oral iron fails.",
    # --- Extended transport biology ---
    "Transferrin receptor 2 (TfR2) is expressed on hepatocytes and erythroblasts and functions as an iron sensor that upregulates hepcidin.",
    "Mutations in TFR2 cause type 3 hereditary hemochromatosis, a juvenile-onset HFE-independent form with inappropriately low hepcidin.",
    "Lactoferrin in breast milk and neutrophil granules sequesters iron, an innate antimicrobial mechanism known as nutritional immunity.",
    "Free (non-transferrin-bound) iron in plasma can drive Fenton chemistry, generating reactive oxygen species that damage tissues in iron overload.",
    "Apotransferrin (iron-free) and holotransferrin (diferric) circulate in plasma; the ratio reflects transferrin saturation.",
    "Diferric transferrin has the highest affinity for TfR1, ensuring iron is preferentially delivered to high-demand cells like erythroblasts.",
    # --- Extended storage biology ---
    "Ferritin is a 24-subunit heteropolymer of H (heavy) and L (light) chains; H-chains have ferroxidase activity, L-chains stabilize storage.",
    "FTH1 (H-chain) gene mutations cause autosomal-dominant adult-onset hyperferritinemia with iron-deficient erythropoiesis.",
    "FTL (L-chain) mutations cause hereditary hyperferritinemia-cataract syndrome via iron-responsive element disruption.",
    "Mitochondrial ferritin (FTMT) is expressed in iron-loaded tissues and may protect mitochondria from iron-induced oxidative damage.",
    "Hemosiderin forms when ferritin is degraded and the iron remains aggregated, visible histologically with Perls Prussian blue stain.",
    "Liver iron concentration (LIC) above 4 mg/g dry weight indicates iron overload; above 15 mg/g raises risk for cirrhosis.",
    "Cardiac iron deposition is best assessed by T2* MRI; values below 20 ms indicate overload, below 10 ms are critical.",
    # --- Extended hepcidin regulation ---
    "BMP6 (bone morphogenetic protein 6) is the principal physiologic ligand activating hepcidin transcription via SMAD1/5/8 signaling.",
    "Hemojuvelin (HJV) is a BMP coreceptor; HJV mutations cause severe juvenile hemochromatosis (type 2A) by abolishing hepcidin expression.",
    "Matriptase-2 (TMPRSS6) cleaves membrane hemojuvelin to dampen BMP signaling and suppress hepcidin transcription.",
    "TfR1 and TfR2 form a complex with HFE on hepatocyte membranes that senses iron-loaded transferrin and modulates hepcidin.",
    "Inflammation-driven hepcidin via IL-6/STAT3 binds the proximal STAT3 binding element in the HAMP promoter.",
    "Activin B is a non-BMP TGF-beta-family ligand that contributes to inflammation-induced hepcidin upregulation.",
    "Erythroferrone (ERFE) is secreted by erythroblasts in response to EPO; it sequesters BMPs and suppresses hepcidin to mobilize iron.",
    "PIEZO1 gain-of-function mutations cause hereditary xerocytosis with chronically suppressed hepcidin and iron overload despite hemolysis.",
    "GDF15 (growth differentiation factor 15) is elevated in beta-thalassemia and suppresses hepcidin, worsening iron loading.",
    "Twisted gastrulation (TWSG1) is a BMP-modulator co-secreted with erythroferrone in thalassemic ineffective erythropoiesis.",
    "Hypoxia stabilizes HIF-2-alpha in enterocytes, which directly upregulates DMT1, DCYTB, and ferroportin transcription.",
    # --- IDA: deeper diagnostic patterns ---
    "Latent iron deficiency precedes anemia: ferritin falls first, then transferrin saturation drops, finally hemoglobin declines.",
    "Mentzer index (MCV / RBC count) below 13 suggests thalassemia trait; above 13 favors iron deficiency anemia.",
    "RDW (red cell distribution width) is typically elevated in IDA reflecting anisocytosis, often normal in thalassemia trait.",
    "Hypochromic, microcytic RBCs on peripheral smear with high RDW are the classic IDA cytologic signature.",
    "Target cells, basophilic stippling, and dimorphic populations should prompt evaluation for thalassemia or sideroblastic anemia.",
    "Iron-restricted erythropoiesis is the broader term encompassing absolute deficiency, functional deficiency (ESA-treated CKD), and ACD.",
    "Functional iron deficiency is defined by adequate stores (ferritin > 100 ng/mL) but TSAT < 20%, common in ESA-treated CKD.",
    "Soluble transferrin receptor / log(ferritin) ratio (sTfR-F index) over 2 favors IDA, under 1 favors ACD.",
    "Reticulocyte hemoglobin equivalent (Ret-He) on Sysmex or CHr on Siemens analyzers tracks iron available for the past 3-4 days of erythropoiesis.",
    "Zinc protoporphyrin (ZPP) accumulates when iron is unavailable for heme synthesis, providing a screening marker in industrial/lead-exposure settings.",
    "Bone marrow iron staining (Perls stain) was the gold standard before non-invasive markers but is rarely needed clinically today.",
    # --- IDA causes (deeper) ---
    "Menstrual blood loss is the leading cause of IDA in pre-menopausal women; menorrhagia warrants gynecologic evaluation.",
    "Colorectal cancer must be ruled out in any iron-deficient adult man or post-menopausal woman without obvious blood loss source.",
    "NSAID-associated gastropathy is a frequently missed cause of chronic occult GI blood loss leading to IDA.",
    "Angiodysplasia of the right colon is a common cause of unexplained IDA in elderly patients, often requires capsule endoscopy.",
    "Cameron lesions are linear erosions on hiatal hernia folds and can cause obscure GI bleeding with IDA.",
    "Gastric antral vascular ectasia (GAVE, watermelon stomach) causes chronic transfusion-dependent IDA and is treated endoscopically with APC.",
    "Hookworm infection (Ancylostoma, Necator) is a major global cause of IDA in tropical regions due to intestinal blood loss.",
    "Cow's milk overconsumption in toddlers causes IDA via low iron content, occult enteric blood loss, and calcium-mediated absorption inhibition.",
    # --- IDA treatment (oral iron deep-dive) ---
    "Ferrous sulfate provides 65 mg elemental iron per 325 mg tablet; ferrous gluconate 36 mg per 325 mg; ferrous fumarate 106 mg per 325 mg.",
    "Heme iron polypeptide is a meat-derived supplement absorbed via HCP1, less affected by hepcidin and food.",
    "Liquid iron preparations (ferrous sulfate elixir) are useful in pediatrics and dysphagia; they may stain teeth and are best taken via straw.",
    "Slow-release iron preparations bypass duodenal absorption sites and generally show inferior absorption to immediate-release ferrous sulfate.",
    "Common oral iron adverse effects include constipation, nausea, dark stools; black stools are not melena unless guaiac-positive.",
    "Vitamin C 200 mg co-administration enhances ferrous absorption 2-3x by maintaining the Fe2+ state and forming soluble chelates.",
    "Coffee, tea, calcium, and antacids should be separated from oral iron doses by at least 2 hours to maximize absorption.",
    "Adequate hemoglobin response is a 1-2 g/dL rise at 2-3 weeks; lack of response prompts evaluation for ongoing loss, malabsorption, or wrong diagnosis.",
    "Oral iron should continue 3-6 months after hemoglobin normalization to replenish ferritin stores (target ferritin > 100 ng/mL).",
    # --- IDA treatment (IV iron) ---
    "Iron sucrose (Venofer) is typically dosed at 100-200 mg per session, multiple sessions needed to deliver a full repletion dose.",
    "Iron dextran (INFeD, DexFerrum) allows total-dose infusion but has the highest rate of severe hypersensitivity reactions.",
    "Ferric carboxymaltose (Injectafer) delivers 750 mg per infusion (up to 1,500 mg total) over weeks, with low anaphylaxis rate.",
    "Ferumoxytol (Feraheme) is a carbohydrate-coated iron oxide originally developed as MRI contrast, now used for iron repletion.",
    "Ferric derisomaltose (Monoferric) allows single 1,000-1,500 mg infusions with infusion-reaction rates similar to ferric carboxymaltose.",
    "Hypophosphatemia is a class effect of IV iron, most pronounced with ferric carboxymaltose due to FGF23 activation; monitor in repeat dosing.",
    "Free-iron infusion reactions present with flushing, hypotension, and back pain; usually resolve with infusion pause and restart at slower rate.",
    "True IgE-mediated iron anaphylaxis is rare with modern formulations; pretreatment with corticosteroids/antihistamines is not routinely recommended.",
    "IV iron is preferred over oral in CKD on ESAs, IBD with active inflammation, post-bariatric, and pre-operative iron repletion windows under 4 weeks.",
    # --- Hereditary Hemochromatosis types ---
    "Type 1 hemochromatosis: HFE mutations (C282Y/C282Y or C282Y/H63D compound heterozygous), most common in Northern European populations.",
    "Type 2A juvenile hemochromatosis: HJV (hemojuvelin) mutations, severe iron overload presenting before age 30 with cardiomyopathy/hypogonadism.",
    "Type 2B juvenile hemochromatosis: HAMP (hepcidin) gene mutations, equally severe phenotype.",
    "Type 3 hemochromatosis: TFR2 (transferrin receptor 2) mutations, intermediate phenotype between type 1 and juvenile forms.",
    "Type 4A hemochromatosis (ferroportin disease, loss-of-function): macrophage iron loading with normal-to-high ferritin and low transferrin saturation.",
    "Type 4B hemochromatosis (ferroportin disease, gain-of-function): hepatocyte iron loading, phenotype resembles HFE hemochromatosis.",
    "African iron overload (Bantu siderosis) was historically attributed to fermented beverages from iron pots but has a polygenic susceptibility component.",
    "Penetrance of C282Y homozygosity is incomplete: only ~10-30% develop clinically significant iron overload, with male sex and alcohol use as modifiers.",
    "Skin hyperpigmentation in HH ('bronze diabetes') is due to combined iron and melanin deposition.",
    "Cardiomyopathy in HH manifests early as diastolic dysfunction and arrhythmias, later as restrictive then dilated cardiomyopathy.",
    "Hypogonadotropic hypogonadism in HH results from pituitary iron deposition impairing LH/FSH secretion, presenting as decreased libido and infertility.",
    "Arthropathy in HH characteristically involves the 2nd and 3rd metacarpophalangeal joints with calcium pyrophosphate deposition.",
    # --- HH diagnostic + treatment depth ---
    "Initial screening for HH: serum ferritin and transferrin saturation; TSAT > 45% is the threshold to proceed with HFE genotyping.",
    "HFE mutation panel typically includes C282Y, H63D, and S65C variants; compound heterozygotes have variable phenotype.",
    "Liver biopsy is reserved for HH patients with ferritin > 1000 ng/mL or elevated transaminases to assess fibrosis stage.",
    "Hepcidin assays are not yet routine in clinical practice but show promise for distinguishing HH subtypes and monitoring chelation.",
    "Initial phlebotomy schedule: weekly 500 mL (200-250 mg iron) until ferritin < 50 ng/mL; then maintenance every 2-4 months.",
    "Therapeutic erythrocytapheresis can remove 2-3x more iron per session than standard phlebotomy and is preferred for severe overload.",
    "Hepatocellular carcinoma risk persists in cirrhotic HH patients even after iron normalization; HCC screening with ultrasound every 6 months.",
    "Family screening for first-degree relatives of C282Y homozygotes is standard; siblings have 25% risk of homozygosity.",
    # --- Iron chelation pharmacology ---
    "Deferoxamine (Desferal) is a hexadentate chelator administered SC over 8-12 hours nightly via pump; binds iron 1:1.",
    "Deferasirox (Exjade, Jadenu) is an oral tridentate chelator dosed once daily; binds iron 2:1 with high specificity.",
    "Deferiprone (Ferriprox) is an oral bidentate chelator dosed three times daily; binds iron 3:1 and uniquely chelates cardiac iron.",
    "Combination deferoxamine + deferiprone is used for severe cardiac iron loading where monotherapy has been insufficient.",
    "Deferasirox can cause renal tubular dysfunction (Fanconi-like syndrome), liver injury, and rarely fatal multi-organ failure.",
    "Deferiprone agranulocytosis affects 1-2% of patients; weekly CBC monitoring is mandatory.",
    "Deferoxamine ototoxicity and retinopathy occur at high doses or in low-iron states; monitor audiology and ophthalmology annually.",
    # --- Anemia of chronic disease (deeper) ---
    "ACD develops in 30-95% of patients with chronic infection, malignancy, or autoimmune disease, depending on the underlying severity.",
    "Inflammatory cytokines (IL-6, IL-1, TNF-alpha, IFN-gamma) collectively drive ACD via hepcidin, EPO blunting, and shortened RBC survival.",
    "ACD typically presents as mild-to-moderate normocytic normochromic anemia with hemoglobin 8-10 g/dL.",
    "Reticulocyte response is inappropriately low in ACD due to EPO blunting and iron sequestration; absolute reticulocyte count < 50,000/uL.",
    "Bone marrow iron stores are normal-to-increased in ACD but iron is sequestered in macrophages, unavailable for erythropoiesis.",
    "ACD coexists with absolute iron deficiency in 20-30% of cases; ferritin > 100 ng/mL excludes pure IDA but does not exclude combined.",
    "Anemia of CKD is a specific form of ACD compounded by decreased EPO production; managed with ESAs and iron repletion to target Hb 10-11 g/dL.",
    "Erythropoiesis-stimulating agents (epoetin alfa, darbepoetin) require adequate iron stores; functional iron deficiency must be corrected.",
    "Anti-hepcidin agents (e.g., LJPC-401, PRS-080) are in development for ACD and beta-thalassemia.",
    "HIF prolyl-hydroxylase inhibitors (roxadustat, vadadustat, daprodustat) stabilize HIF-2 to increase endogenous EPO and improve iron utilization in CKD anemia.",
    # --- IRIDA (deeper) ---
    "IRIDA prevalence is unknown; case series suggest under-diagnosis because oral iron failure is often attributed to non-compliance.",
    "TMPRSS6 mutations cause loss of matriptase-2 function, leading to inappropriately high hepcidin despite iron deficiency.",
    "IRIDA labs: low MCV, low MCH, low TSAT, normal-to-high ferritin paradoxically (hepcidin-driven sequestration), elevated hepcidin level if measured.",
    "Carrier (heterozygous) TMPRSS6 variants may contribute to suboptimal response to oral iron in otherwise unexplained IDA.",
    "IRIDA does not always require chronic IV iron — some patients reach adequate Hb but never normalize MCV or fully replete stores.",
    "Distinguishing IRIDA from ACD: IRIDA has low TSAT and microcytosis (mismatched to ferritin); ACD typically has low MCV inflammation context.",
    # --- Sideroblastic anemia + extras ---
    "Sideroblastic anemia features mitochondrial iron loading visible as ring sideroblasts on Perls-stained bone marrow.",
    "X-linked sideroblastic anemia from ALAS2 mutations may respond to pyridoxine (vitamin B6) supplementation.",
    "Acquired clonal sideroblastic anemia (MDS-RS) frequently harbors SF3B1 splicing mutations.",
    "Drug-induced sideroblastic anemia is caused by isoniazid, chloramphenicol, linezolid, and lead poisoning.",
    "Erythropoietic protoporphyria features impaired heme synthesis and presents with photosensitivity, not typically anemia.",
    # --- Thalassemia interactions ---
    "Beta-thalassemia major patients become transfusion-dependent in early childhood; iron overload from chronic transfusion is the leading cause of death.",
    "Non-transfusion-dependent thalassemia (NTDT) develops iron overload primarily from increased absorption due to suppressed hepcidin.",
    "Ineffective erythropoiesis in thalassemia drives erythroferrone secretion, suppressing hepcidin and exacerbating iron loading.",
    "Luspatercept is an activin receptor ligand trap that reduces erythroferrone and improves anemia in beta-thalassemia and MDS-RS.",
    # --- Pregnancy ---
    "Plasma volume expands by 40-50% during pregnancy, causing dilutional anemia even with adequate iron stores.",
    "Iron requirements rise to ~1000 mg total over pregnancy: fetal/placental needs, plasma expansion, blood loss at delivery.",
    "First-trimester ferritin under 30 ng/mL predicts third-trimester IDA without prophylactic iron.",
    "Routine antenatal iron supplementation (30-60 mg elemental iron daily) reduces maternal anemia and improves birthweight.",
    "Postpartum IV iron is appropriate for moderate-severe anemia (Hb < 9.5 g/dL) when oral iron is not tolerated or response is needed quickly.",
    "Maternal iron deficiency in pregnancy is associated with preterm birth, low birth weight, and impaired infant cognitive development.",
    "Cord blood ferritin reflects fetal iron stores; values under 75 ug/L are associated with developmental delay through 5 years.",
    # --- Pediatric ---
    "Infants are born with ~75 mg/kg of iron, most in hemoglobin; iron stores deplete by 6 months without dietary iron.",
    "Exclusively breastfed infants beyond 6 months are at risk for IDA; AAP recommends iron-fortified cereals or supplementation.",
    "Cow's milk should be limited to 24 oz/day in toddlers due to low iron and inhibition of non-heme iron absorption.",
    "Iron-deficiency anemia in early childhood is associated with measurable, sometimes irreversible, cognitive and motor delays.",
    "Lead poisoning often coexists with iron deficiency; both interfere with heme synthesis and have overlapping presentations.",
    # --- Athletic / sports anemia ---
    "Foot-strike hemolysis in distance runners causes mild hemolytic anemia and chronic iron loss via hematuria and hemoglobinuria.",
    "Exercise-induced hepcidin elevation (peak 3-6 hours post-strenuous exercise) impairs iron absorption in athletes.",
    "Endurance athletes should consume iron supplements in the morning before training to maximize absorption.",
    "Hemoglobin masking from plasma volume expansion can be mistaken for sports anemia; ferritin and TSAT clarify true iron status.",
    # --- CKD / dialysis ---
    "Iron losses in hemodialysis average 1-2 g/year via dialyzer retention, blood sampling, and access bleeding.",
    "KDIGO targets for iron sufficiency in CKD: ferritin > 100 ng/mL (HD: > 200 ng/mL) and TSAT > 20%.",
    "Iron repletion lowers ESA requirements in dialysis, reducing cardiovascular adverse-event exposure.",
    "Ferric pyrophosphate citrate (Triferic) is administered via dialysate to maintain iron balance without IV access.",
    # --- Iron + immunity / infection ---
    "Withholding iron from pathogens (nutritional immunity) is mediated by hepcidin, lactoferrin, and lipocalin-2.",
    "Iron repletion increases susceptibility to Yersinia, Vibrio, Salmonella, and Listeria infections; caution in untreated bacteremia.",
    "Iron loading worsens malaria progression; chelators have shown adjunctive antimalarial activity in experimental models.",
    # --- Iron + cancer ---
    "Iron-overload states (HH, transfusional siderosis) increase risk of hepatocellular carcinoma 20-200x in cirrhotic patients.",
    "Cancer cells often upregulate TfR1 to meet high proliferation iron demand, a target for radiolabeled therapy and imaging.",
    "Anemia in cancer is multifactorial: marrow infiltration, chemotherapy myelosuppression, occult bleeding, and ACD from cytokine release.",
    # --- Iron + neurodegeneration ---
    "Brain iron accumulates with age, especially in basal ganglia; excess deposition is implicated in Parkinson's and Alzheimer's pathology.",
    "Neurodegeneration with brain iron accumulation (NBIA) is a group of genetic disorders including PKAN (PANK2) and BPAN (WDR45).",
    "Friedreich ataxia involves mitochondrial iron mishandling due to frataxin deficiency, with secondary cardiomyopathy.",
    "Restless legs syndrome is associated with low brain iron despite normal systemic iron; ferritin under 75 ng/mL warrants supplementation trial.",
    # --- Iron sulfur clusters ---
    "Iron-sulfur (Fe-S) clusters are essential cofactors in mitochondrial electron transport, DNA repair, and ribosome assembly.",
    "ISCU mutations cause hereditary myopathy with lactic acidosis due to defective Fe-S cluster assembly.",
    "Cytosolic Fe-S clusters require ABCB7 export from mitochondria; ABCB7 mutations cause X-linked sideroblastic anemia with ataxia.",
    # --- Heme synthesis / porphyrias ---
    "Heme biosynthesis spans mitochondria and cytosol, with rate-limiting ALA synthase in mitochondria.",
    "Porphyria cutanea tarda is the most common porphyria, associated with hereditary or acquired iron overload (HFE, alcohol, HCV).",
    "Phlebotomy is first-line treatment for PCT, reducing hepatic iron and restoring uroporphyrinogen decarboxylase activity.",
    # --- Diagnostic algorithms / pitfalls ---
    "In suspected combined IDA + ACD, a 1-2 month oral iron trial with reticulocyte response is diagnostic and therapeutic.",
    "Ferritin > 1000 ng/mL with normal TSAT suggests hyperferritinemia syndromes: still-disease, MAS, GAH, FTL mutations, or non-iron etiologies.",
    "Hyperferritinemia hemophagocytic lymphohistiocytosis (HLH): ferritin > 10,000 ng/mL combined with cytopenias is highly suggestive.",
    "Genetic hyperferritinemia (FTL IRE mutations) typically presents with isolated very high ferritin and bilateral early cataracts.",
    # --- Treatment monitoring ---
    "Hemoglobin should be checked 2-4 weeks after starting oral iron; failure prompts adherence, GI loss, and IRIDA evaluation.",
    "Ferritin and TSAT should be rechecked 3 months after IV iron course; over-replacement carries hepatic iron deposition risk.",
    "Hepcidin response to iron challenge (post-oral-dose hepcidin) can distinguish hepcidin-driven malabsorption (IRIDA) from absolute deficiency.",
    # --- Pharmacovigilance / safety ---
    "Iron poisoning in pediatric ingestion: stages include initial GI hemorrhage, latent phase, systemic toxicity, hepatic failure, late stricture.",
    "Deferoxamine is the chelator of choice for acute iron toxicity; gastric decontamination is largely abandoned for iron ingestion.",
    "Multivitamin pediatric iron content is low; serious toxicity usually requires adult-strength prenatal vitamins or pure iron supplements.",
    # --- Lab nuances ---
    "Hemolyzed blood samples falsely elevate serum iron; recollect if hemolysis index is high.",
    "Fasting morning samples are preferred for serum iron and TSAT to minimize diurnal variation.",
    "Ferritin can be falsely lowered in liver failure (decreased synthesis) and falsely elevated in chronic alcohol use.",
    "Erythrocyte ferritin (intra-RBC ferritin) tracks long-term iron status (RBC lifespan) and is unaffected by acute inflammation.",
    # --- Special diets ---
    "Vegetarian and vegan diets require 1.8x the iron RDA due to lower non-heme bioavailability.",
    "Combining vitamin C-rich foods with plant iron sources is the most practical bioavailability hack for non-heme iron.",
    "Cast-iron cookware contributes meaningful iron in acidic preparations (tomato sauce) — historically protective in some populations.",
    "Iron fortification of wheat flour and infant formula has substantially reduced national IDA rates in many countries.",
    # --- Edge cases ---
    "Transient erythroblastopenia of childhood causes transient anemia without iron deficiency; ferritin and reticulocytes guide diagnosis.",
    "Aceruloplasminemia: rare loss of plasma ceruloplasmin ferroxidase causes brain and pancreatic iron loading with neurodegeneration and diabetes.",
    "Atransferrinemia: rare TF gene mutations cause severe IDA with paradoxical hepatic and cardiac iron overload from non-transferrin-bound iron.",
    "DMT1 mutations (rare): cause hypochromic microcytic anemia with hepatic iron loading; oral iron is ineffective, transfusions worsen overload.",
    # --- Imaging refinements ---
    "MRI T2* values: > 20 ms is normal cardiac iron, 10-20 ms is mild-moderate overload, < 10 ms is severe risk for cardiomyopathy.",
    "Hepatic R2* MRI correlates with LIC and is the non-invasive standard for monitoring chelation response.",
    "FerriScan (Resonance Health R2 MRI) provides FDA-cleared quantitative LIC measurement.",
    # --- Research frontiers ---
    "Mini-hepcidins are synthetic short peptides mimicking endogenous hepcidin, in trials for HH and iron-loading anemias.",
    "Erythroferrone-neutralizing antibodies are an experimental approach to restore hepcidin in beta-thalassemia.",
    "Vamifeport is an oral ferroportin inhibitor that mimics hepcidin's action, in clinical trials for thalassemia and polycythemia vera.",
    "TMPRSS6-ASO and TMPRSS6-siRNA approaches aim to raise hepcidin in iron-loading anemias by suppressing matriptase-2.",
    # --- Public health ---
    "WHO estimates 30% of women and 40% of preschool children worldwide have anemia, most attributable to iron deficiency.",
    "Iron fortification has been adopted in over 80 countries; salt iron fortification is the newest large-scale strategy.",
    "Mass deworming campaigns combined with iron supplementation are effective in hookworm-endemic regions.",
    # --- Operative / preoperative ---
    "Preoperative anemia is independently associated with increased perioperative mortality; correction is recommended ≥ 4 weeks before elective surgery.",
    "Patient blood management programs combine preoperative iron repletion, intraoperative cell salvage, and restrictive transfusion thresholds.",
    "Tranexamic acid reduces surgical bleeding and the need for transfusion, indirectly preserving iron stores.",
]


def _drop_table_if_exists(adb_cfg, table_name: str) -> None:
    conn = oracledb.connect(
        user=adb_cfg["user"],
        password=adb_cfg["password"],
        dsn=adb_cfg["dsn"],
        config_dir=adb_cfg["wallet_location"],
        wallet_location=adb_cfg["wallet_location"],
        wallet_password=adb_cfg["wallet_password"],
    )
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f"DROP TABLE {table_name} PURGE")
                conn.commit()
            except oracledb.DatabaseError as exc:
                if "ORA-00942" not in str(exc):
                    raise
    finally:
        conn.close()


def _seed_table_sync(
    adb_cfg, table_name: str, dim: int, embeddings: list[list[float]], docs: list[str]
) -> None:
    conn = oracledb.connect(
        user=adb_cfg["user"],
        password=adb_cfg["password"],
        dsn=adb_cfg["dsn"],
        config_dir=adb_cfg["wallet_location"],
        wallet_location=adb_cfg["wallet_location"],
        wallet_password=adb_cfg["wallet_password"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE {table_name} (
                    id VARCHAR2(255) PRIMARY KEY,
                    content CLOB,
                    embedding VECTOR({dim}, FLOAT32),
                    metadata CLOB CHECK (metadata IS JSON),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
                )
                """
            )
            for i, (text, emb) in enumerate(zip(docs, embeddings, strict=True)):
                vec = "[" + ",".join(str(f) for f in emb) + "]"
                cur.execute(
                    f"INSERT INTO {table_name} (id, content, embedding, metadata) VALUES (:id, :c, TO_VECTOR(:e), :m)",  # noqa: S608
                    {
                        "id": f"doc-{i:02d}",
                        "c": text,
                        "e": vec,
                        "m": json.dumps({"topic": "iron_metabolism"}),
                    },
                )
            conn.commit()
    finally:
        conn.close()


async def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    adb_cfg = {
        "dsn": os.environ["ADB_DSN"],
        # Default to a least-priv app schema (locus_app), not ADMIN.
        # See docs/concepts/rag.md for the CREATE USER / GRANT script.
        "user": os.environ.get("ADB_USER", "locus_app"),
        "password": os.environ["ADB_PASSWORD"],
        "wallet_location": os.path.expanduser(os.environ["ADB_WALLET_LOCATION"]),
        "wallet_password": os.environ.get("ADB_WALLET_PASSWORD", os.environ["ADB_PASSWORD"]),
    }
    oci_profile = os.environ["OCI_PROFILE"]
    compartment = os.environ["OCI_COMPARTMENT"]
    endpoint = os.environ.get(
        "OCI_ENDPOINT",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    # gpt-5.1 is the highest-capacity reliable tool-caller on OCI right now;
    # Gemini 2.5 Pro is currently returning empty completions for
    # tool-augmented prompts (service-side issue, May 16 2026).
    model_id = os.environ.get("OCI_RESEARCH_MODEL", "oci:openai.gpt-5.1")
    max_out = int(os.environ.get("MAX_OUTPUT_TOKENS", "65536"))

    table_name = f"LOCUS_IRON_{uuid.uuid4().hex[:8].upper()}"
    print(f"== locus deep-research replay ==")
    print(f"   model         : {model_id}")
    print(f"   max_out_tokens: {max_out}")
    print(f"   ADB           : {adb_cfg['dsn']} table={table_name}")
    print(f"   corpus size   : {len(CORPUS)} sentences")
    print()

    # 1. Build embedder, embed the whole corpus, write to ADB
    print("[1/3] Embedding corpus with cohere.embed-v4.0 …")
    embedder = OCIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=compartment,
        profile_name=oci_profile,
        auth_type=os.environ.get("OCI_AUTH_TYPE", "api_key"),
        service_endpoint=endpoint,
    )
    embs = await embedder.embed_documents(CORPUS)
    raw_embs = [e.embedding for e in embs]
    dim = len(raw_embs[0])
    print(f"       embeddings: {len(raw_embs)} × {dim}-d")

    try:
        print(f"[1.5/3] Creating + seeding table {table_name} …")
        _seed_table_sync(adb_cfg, table_name, dim, raw_embs, CORPUS)

        # 2. Wire OracleVectorStore + RAGRetriever; hand to create_deepagent
        print(f"[2/3] Building deepagent with datastores={{medical: <ADB>}} …")
        store = OracleVectorStore(
            **adb_cfg,
            table_name=table_name,
            dimension=dim,
            auto_create_table=False,
        )
        retriever = RAGRetriever(embedder=embedder, store=store)

        chat = get_model(
            model_id,
            profile=oci_profile,
            compartment_id=compartment,
            region=region,
        )
        agent = create_deepagent(
            model=chat,
            system_prompt=(
                "You are a research-grade medical writer. When asked to "
                "produce a memo on a hematology topic, FIRST call the "
                "search_medical tool with at least two distinct queries "
                "that together cover the topic, then synthesize a "
                "long-form memo with clear section headers, bullet "
                "points where they aid clarity, and inline citations "
                "of the form (doc-NN) pointing back to the retrieved "
                "documents. Aim for a 1,500-3,000 word memo. Do not "
                "fabricate facts beyond what the retrieved documents "
                "support; if a claim isn't in the retrieved set, omit "
                "it or flag it as 'not in corpus'."
            ),
            tools=[],
            datastores={
                "medical": {
                    "retriever": retriever,
                    "description": (
                        "comprehensive hematology corpus on iron metabolism — "
                        "absorption (heme, non-heme, DMT1, HCP1, DCYTB, "
                        "hephaestin), transport (transferrin, TfR1, TfR2, "
                        "sTfR), storage (ferritin, hemosiderin), regulation "
                        "(hepcidin / ferroportin axis, BMP6/SMAD, IL-6/STAT3, "
                        "erythroferrone, HFE, hemojuvelin, TMPRSS6), "
                        "disorders (IDA, hereditary hemochromatosis types 1-4, "
                        "anemia of chronic disease, IRIDA, sideroblastic, "
                        "thalassemia, ferroportin disease, transfusional "
                        "siderosis), diagnostics (lab markers, MRI T2*, "
                        "genetics), treatments (oral iron formulations, IV "
                        "iron formulations, chelators, phlebotomy, novel "
                        "agents), and special populations (pregnancy, "
                        "pediatric, CKD, athletic, geriatric, post-bariatric)."
                    ),
                    "top_k": 10,
                }
            },
            max_output_tokens=max_out,
            max_iterations=25,
            # The default total-run TokenLimit (max_tokens=80_000) can fire
            # before a 65K-output memo finishes; bump it well past 65K + a
            # few tool-call round trips.
            max_tokens=200_000,
            reflexion=False,
            grounding=False,
        )

        # 3. Run the real research question + save artifacts.
        # We iterate the async event stream so we can print each tool call's
        # returned snippets live — that's the evidence the agent is actually
        # consuming retrieved DB rows rather than hallucinating.
        print("[3/3] Running research turn (gpt-5.1 @ 65K may take 60-300s) …")
        from locus.core.events import (
            ModelCompleteEvent,
            TerminateEvent,
            ToolCompleteEvent,
            ToolStartEvent,
        )

        t0 = time.time()
        args_by_id: dict[str, dict] = {}
        # Each entry: (tool_name, args, full_result, duration_ms, error)
        tool_records: list[tuple[str, dict, str, float | None, str | None]] = []
        text = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        async for event in agent.run(
            "Write an exhaustive, textbook-grade research memo on iron "
            "metabolism and its clinical significance. Aim for 50,000-60,000 "
            "characters (roughly 8,000-10,000 words). Search the medical "
            "datastore aggressively — issue at least 15 distinct search "
            "queries that together cover ALL of the following sections in "
            "depth, with extensive sub-sections under each:\n\n"
            "1. Dietary iron and absorption mechanisms (heme vs non-heme, "
            "DMT1, HCP1, DCYTB, hephaestin, dietary modifiers, drugs that "
            "impair absorption, anatomic factors, GI conditions affecting "
            "absorption).\n"
            "2. Plasma transport (transferrin, TfR1, TfR2, sTfR, "
            "ceruloplasmin/hephaestin, holotransferrin vs apotransferrin).\n"
            "3. Storage (ferritin biology — H/L chains, FTL/FTH1 mutations, "
            "mitochondrial ferritin, hemosiderin, body iron distribution).\n"
            "4. Cellular uptake and iron-sulfur cluster biogenesis.\n"
            "5. Hepcidin–ferroportin axis in detail: BMP6/SMAD, IL-6/STAT3, "
            "erythroferrone, HJV, TMPRSS6, HFE, TfR2.\n"
            "6. Iron deficiency anemia: epidemiology, pathophysiology, all "
            "causes (menstrual, GI, dietary, post-bariatric, H. pylori, "
            "celiac, hookworm), diagnostic patterns, oral iron formulations "
            "with elemental iron content, IV iron formulations with dosing "
            "and adverse events, monitoring.\n"
            "7. Hereditary hemochromatosis — every type (1, 2A, 2B, 3, 4A, "
            "4B), pathophysiology, organ-specific manifestations, diagnostic "
            "algorithm, phlebotomy and chelation regimens, family screening.\n"
            "8. Iron chelators (deferoxamine, deferasirox, deferiprone): "
            "binding stoichiometry, dosing, adverse events, combinations.\n"
            "9. Anemia of chronic disease: inflammatory cytokines, hepcidin, "
            "ESA dependency, CKD anemia, HIF-PHI agents, novel anti-hepcidin "
            "agents in development.\n"
            "10. IRIDA — TMPRSS6 biology, lab features, distinguishing from "
            "ACD and IDA, treatment approach.\n"
            "11. Sideroblastic anemias (X-linked, acquired clonal, "
            "drug-induced).\n"
            "12. Thalassemia interactions, ineffective erythropoiesis, "
            "luspatercept.\n"
            "13. Special populations: pregnancy, pediatric, athletes, CKD, "
            "geriatric.\n"
            "14. Iron and immunity, infection, cancer, neurodegeneration.\n"
            "15. Imaging (MRI T2*, R2*, FerriScan).\n"
            "16. Lab nuances, diagnostic algorithms, treatment monitoring.\n"
            "17. Research frontiers (mini-hepcidins, erythroferrone "
            "antibodies, vamifeport, TMPRSS6 ASOs).\n"
            "18. Public health and operative considerations.\n\n"
            "For every clinical claim, cite at least one (doc-NN) supporting "
            "document. Use markdown section headers, sub-headers, bullets, "
            "and tables where they aid clarity. Conclude with a "
            "comprehensive 'Take-Home Points' section. Do NOT invent facts; "
            "if a claim is not in the retrieved corpus, omit it or flag "
            "'not in corpus'."
        ):
            if isinstance(event, ToolStartEvent):
                args_by_id[event.tool_call_id] = event.arguments or {}
                q = (event.arguments or {}).get("query", "")
                print(f"\n>>> search call: {event.tool_name}({q!r})")
            elif isinstance(event, ToolCompleteEvent):
                args = args_by_id.get(event.tool_call_id, {})
                result_str = str(event.result or "")
                tool_records.append(
                    (event.tool_name, args, result_str, event.duration_ms, event.error)
                )
                # Print first 600 chars of what came back from the DB so the
                # operator can see real evidence flowing into the model.
                preview = result_str.replace("\n", " ")[:600]
                print(
                    f"    returned {len(result_str)} chars ({event.duration_ms:.0f}ms)"
                    + (f" ERROR={event.error}" if event.error else "")
                )
                print(f"    preview: {preview}{'…' if len(result_str) > 600 else ''}")
            elif isinstance(event, ModelCompleteEvent):
                usage = event.usage or {}
                total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            elif isinstance(event, TerminateEvent):
                text = event.final_message or ""
        elapsed = time.time() - t0

        # Build a result-like object so the persisted trace JSON stays the
        # same shape the user is used to from prior runs.
        class _M:
            iterations = None
            tool_calls = len(tool_records)
            total_tokens = total_prompt_tokens + total_completion_tokens
            prompt_tokens = total_prompt_tokens
            completion_tokens = total_completion_tokens
            duration_ms = elapsed * 1000

        class _T:
            def __init__(self, name, args, result_str, duration_ms, error):
                self.tool_name = name
                self.arguments = args
                self.result = result_str
                self.duration_ms = duration_ms
                self.error = error

        tool_execs = [_T(*r) for r in tool_records]
        metrics = _M()

        # Persist the memo + trace
        memo_path = out_dir / "iron_metabolism_memo.md"
        memo_path.write_text(text)
        trace_path = out_dir / "iron_metabolism_trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "model": model_id,
                    "max_output_tokens": max_out,
                    "elapsed_seconds": elapsed,
                    "corpus_size": len(CORPUS),
                    "metrics": {
                        "iterations": getattr(metrics, "iterations", None),
                        "tool_calls": getattr(metrics, "tool_calls", None),
                        "total_tokens": getattr(metrics, "total_tokens", None),
                        "prompt_tokens": getattr(metrics, "prompt_tokens", None),
                        "completion_tokens": getattr(metrics, "completion_tokens", None),
                        "duration_ms": getattr(metrics, "duration_ms", None),
                    },
                    "tool_calls": [
                        {
                            "name": t.tool_name,
                            "arguments": t.arguments,
                            "result_chars": len(t.result or ""),
                            "error": t.error,
                            "duration_ms": t.duration_ms,
                        }
                        for t in tool_execs
                    ],
                },
                indent=2,
            )
        )

        # Summary on stdout
        words = len(text.split())
        chars = len(text)
        # Citation density: which doc-NN ids appear in the memo, and how
        # many times each. This is the strongest evidence the agent
        # actually consumed the retrieved snippets.
        import re

        cites_all = re.findall(r"doc-(\d{2})", text)
        unique_cites = sorted(set(cites_all))
        print()
        print(f"-- run complete in {elapsed:.1f}s --")
        if metrics:
            print(f"   tool calls         : {metrics.tool_calls}")
            print(
                f"   tokens             : prompt={metrics.prompt_tokens} "
                f"completion={metrics.completion_tokens} total={metrics.total_tokens}"
            )
        print(f"   memo               : {words} words, {chars:,} chars")
        print(f"   citations (raw)    : {len(cites_all)}")
        print(f"   citations (unique) : {len(unique_cites)} / {len(CORPUS)} docs in corpus")
        if unique_cites:
            print(f"   docs cited         : {', '.join('doc-' + c for c in unique_cites)}")
        retrieved_total = sum(len(r.result or "") for r in tool_execs)
        print(
            f"   retrieved (total)  : {retrieved_total:,} chars across {len(tool_execs)} search calls"
        )
        print(f"   saved              : {memo_path}")
        print(f"   trace              : {trace_path}")

        try:
            await store.close()
        except BaseException:
            pass
    finally:
        _drop_table_if_exists(adb_cfg, table_name)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
