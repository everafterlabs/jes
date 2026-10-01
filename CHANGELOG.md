# Changelog

All notable changes to jes will be documented here.

The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- A missing `TYPESAFE_API_KEY` raises `BackendError` with reason `missing_api_key`, not `client_setup_failed`.
- A failed hook check prints the backend's reason, and says to run `jes login` when the API key is missing.

## 0.0.1

First release.
