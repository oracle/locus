---
name: api-design
description: Use this skill when designing REST APIs or reviewing API endpoints. Provides best practices for consistent, developer-friendly APIs.
allowed-tools: write_file read_file
metadata:
  author: oracle
  version: "1.0"
---

# REST API Design Best Practices

## URL Structure

- Use nouns, not verbs: `/users` not `/getUsers`
- Use plural: `/orders` not `/order`
- Nest for relationships: `/users/{id}/orders`
- Max 3 levels of nesting

## HTTP Methods

- GET: Read (idempotent, no body)
- POST: Create (returns 201 + Location header)
- PUT: Full update (idempotent)
- PATCH: Partial update
- DELETE: Remove (returns 204)

## Response Format

- Always return JSON with consistent structure
- Include `data`, `error`, `meta` top-level keys
- Paginate collections: `?page=1&limit=20`
- Return total count in meta for pagination

## Error Handling

- Use standard HTTP status codes
- 400: Bad request (validation failed)
- 401: Unauthorized (no/invalid auth)
- 403: Forbidden (valid auth, no permission)
- 404: Not found
- 409: Conflict (duplicate resource)
- 500: Server error (never expose internals)

## Versioning

- Use URL path versioning: `/v1/users`
- Never break existing clients
- Deprecate with headers before removing
