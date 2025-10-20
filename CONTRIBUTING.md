## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples:
```bash
git commit -m "feat: add support for DuckDB backend"
git commit -m "fix: resolve memory leak in scoring module"
git commit -m "docs: update installation instructions"
git commit -m "chore: update dependencies to latest versions"
```

### Breaking Changes:
```bash
git commit -m "feat!: remove deprecated API endpoints

BREAKING CHANGE: The /v1/score endpoint has been removed.
Use /v2/score instead."
```