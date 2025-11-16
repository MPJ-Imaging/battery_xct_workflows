# Contributing

Contributions are welcome, especially:

- Improvements or fixes to existing notebooks and utilities.
- New analysis workflows or examples.
- New or improved ML models under `models/`.

## How to contribute

1. **Open an issue** describing the change you’d like to make (bug, feature, or improvement).
2. **Fork the repo** and create a feature branch from `dev` (or as documented in the README).
3. Make your changes, keeping:
   - Notebooks in `notebooks/`
   - Reusable Python code in `utils/`
   - Model-related code in `models/`
4. **Add or update tests** in `models/tests` if you change model code.
5. Ensure:
   - `pytest` passes
   - The core notebooks still run (or at least your changes don’t obviously break them).
6. **Open a pull request** referencing the issue and summarising your changes.

By submitting a contribution, you agree that your work will be released under this project’s license.
