# Terraform Draft

This folder is an Azure infrastructure draft for the RAG knowledge base. It is
included to show the intended deployment direction, but it is not a complete
production deployment yet.

## What Is Included

- Azure resource group.
- Azure Container Registry.
- Linux App Service Plan.
- Linux Web App placeholder.

## What Is Missing Before Production

- Container image configuration for the Web App.
- Secure secret management through Key Vault or managed app settings.
- Managed identity and least-privilege role assignments.
- Network restrictions, HTTPS-only settings, and access controls.
- Deployment outputs and environment-specific variables.
- Persistent storage strategy for uploaded PDFs and vector indexes.
- Monitoring, alerting, and log retention settings.

## Intended Next Step

Complete this module only after the local Docker Compose workflow and CI are
stable. The production version should deploy a built container image, inject
`GOOGLE_API_KEY` and optional `RAG_API_TOKEN` securely, and use managed storage
for documents and vector data.
