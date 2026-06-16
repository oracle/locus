---
name: code-review
description: Use this skill when reviewing code for quality, security, and maintainability issues. Provides a structured checklist for thorough code reviews.
allowed-tools: read_file search_code
metadata:
  author: oracle
  version: "1.0"
---

# Code Review Checklist

## 1. Security

- Check for hardcoded secrets (API keys, passwords, tokens)
- Validate all user inputs before use
- Check for SQL injection, XSS, command injection
- Ensure sensitive data is not logged

## 2. Error Handling

- All external calls wrapped in try/except
- Errors logged with context (not just swallowed)
- User-facing errors are safe (no stack traces leaked)

## 3. Code Quality

- Functions are under 50 lines
- No duplicated logic (DRY principle)
- Clear variable and function names
- Type hints on public functions

## 4. Testing

- New code has corresponding tests
- Edge cases covered (empty input, None, large data)
- Tests are independent (no shared mutable state)

## 5. Performance

- No N+1 queries
- Large collections use generators, not lists
- Expensive operations are cached where appropriate

## Summary Format

After reviewing, provide:

1. **Critical issues** (must fix before merge)
2. **Suggestions** (improve quality but not blocking)
3. **Positives** (what was done well)
