# 1. definiere Azure als mein Cloud-Provider
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# 2. Ressourcengruppe (Der logische Ordner in der Cloud)
resource "azurerm_resource_group" "rg" {
  name     = "rag-knowledge-base-rg"
  location = "West Europe"
}

# 3. Container Registry (Der private Hafen für mein fertiges Docker-Image)
resource "azurerm_container_registry" "acr" {
  name                = "ragknowledgebaseacr" # Muss weltweit einzigartig sein
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

# 4. App Service Plan (Der eigentliche Server/die CPU)
resource "azurerm_service_plan" "app_plan" {
  name                = "rag-app-plan"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  os_type             = "Linux"
  sku_name            = "B1"
}

# 5. Web App (Hier läuft am Ende mein Docker-Container)
resource "azurerm_linux_web_app" "app" {
  name                = "rag-knowledge-base-app"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  service_plan_id     = azurerm_service_plan.app_plan.id

  site_config {
    always_on = false
  }
}
