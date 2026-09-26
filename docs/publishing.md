# Publishing workflow

This guide describes how changes are prepared, reviewed and published.

## Branches

- `main` always contains working, documented content.
- Develop each change on a short-lived branch named `<type>/<short-description>`, where the type is `feature`, `fix`, `docs` or `chore`; for example, `feature/heston-calibration`.

## Commits

- Write commit messages in English, in the imperative mood, with a concise summary line of no more than about 72 characters: `Add Heston calibration`, `Fix day-count convention in bond pricer`.
- Keep each commit focused on a single logical change.
- Before committing, review `git status` and `git diff`, and stage only the files relevant to the change.

## Pull requests

- Merge into `main` through a pull request that explains what changed, why, and how it was verified.
- Report the commands actually run and their outcomes, and flag any check that could not be performed.

## Pre-publication checklist

- [ ] No API keys, tokens, passwords, personal data, confidential data or session transcripts are included.
- [ ] Credentials are read from environment variables, and any `.env.example` file contains placeholder values only.
- [ ] Notebooks have been restarted and run from top to bottom, and sensitive or oversized outputs have been cleared.
- [ ] The project README is complete and the catalogue in the main README is up to date.
- [ ] Reused code is credited and compatible with the MIT License.
