# Security Policy

## Reporting a Vulnerability

Do not open a public issue for suspected vulnerabilities or exposed secrets.
Use the repository's private vulnerability reporting form instead:

https://github.com/idevelopAI/genai-knowledge-base/security/advisories/new

Include the affected component, reproduction steps, impact, and any suggested
mitigation. Reports will be reviewed before details are disclosed publicly.

## Secrets

Never include real API keys, bearer tokens, `.env` files, uploaded internal
documents, or generated vector-store contents in issues or pull requests.
Rotate a credential immediately if it may have been exposed.

## Scope

The default deployment is intended for local, single-instance use. The
documented limitations around local storage, in-memory controls, and the draft
Terraform configuration should be considered when assessing deployment risk.
