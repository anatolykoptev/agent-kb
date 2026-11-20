# Contributing

1. Open an issue for anything beyond a small fix.
2. Run `tests/run.sh` before pushing (PostgreSQL 18 with pgvector on 127.0.0.1:5432; set `KB_REQUIRE_PG=1` to fail instead of skip). CI runs it on `ubuntu-24.04-arm`.
3. Add a test that fails without your change. A change to the deny rules needs a test per rule.
4. One logical change per pull request; describe what changed, why, and how it was tested.

Security issues: see [SECURITY.md](SECURITY.md), not a public issue.
