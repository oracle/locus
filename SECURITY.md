# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in Tulip, please report it to us through coordinated disclosure.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use [GitHub private vulnerability reporting](https://github.com/tuliplabs-ai/tulip-agents/security/advisories/new) on this repository, including:

- A description of the vulnerability
- Steps to reproduce the issue
- Any potential impact
- Any suggested fixes (if applicable)

We will acknowledge receipt of your vulnerability report and send you regular updates about our progress.

## Supported Versions

Security fixes land on the **latest released minor of the current major**.
Older minors are not backported — upgrade within the major instead, which
[`DEPRECATION.md`](DEPRECATION.md) guarantees is non-breaking.

| Version                       | Supported          |
| ----------------------------- | ------------------ |
| Latest `2.x` minor            | :white_check_mark: |
| Earlier `2.x` minors          | :x: — upgrade to the latest `2.x` |
| `< 2.0`                       | :x:                |

Pin a major, as in `tulip-agents>=2,<3`, and take the newest minor within it.
Minors ship often, so an enumerated list of supported minors would be stale
faster than it could be maintained; this table is written to stay correct
without edits.

## Security Best Practices

When using the SDK in production:

1. **API Keys**: Never commit API keys or secrets to version control. Use environment variables or secret management systems.

2. **Tool Execution**: Be cautious when allowing agents to execute tools that interact with external systems. Implement proper sandboxing and validation.

3. **Input Validation**: Always validate and sanitize user inputs before passing them to agents.

4. **Model Access**: Use appropriate IAM policies and scoped API keys to restrict access to your model providers.

5. **Checkpointing**: When using persistent checkpointing backends (Redis, PostgreSQL, etc.), ensure proper authentication and encryption in transit.
