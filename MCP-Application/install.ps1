# MCP Server Environment Quickstart Setup Script
# ==============================================
#
# This script provides automated environment setup for MCP (Model Context Protocol) server development.
# It focuses purely on setting up the development environment and dependencies with intelligent detection.

# Set execution policy for this process
try {
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
    Write-Host "Execution Policy set to Bypass for current process"
} catch {
    Write-Warning "Failed to set execution policy: $_"
}

# Self-elevate the script if required
if (-Not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] 'Administrator')) {
    Write-Warning "Script requires Admin privileges. Attempting to elevate..."
    $CommandLine = "-ExecutionPolicy Bypass -File `"" + $MyInvocation.MyCommand.Path + "`" " + $MyInvocation.UnboundArguments
    Start-Process -FilePath PowerShell.exe -Verb RunAs -ArgumentList $CommandLine
    Exit
}

# Set error action preference
$ErrorActionPreference = "Stop"

# ANSI color codes for Windows PowerShell
$Colors = @{
    Green = "$([char]0x1b)[92m"
    Yellow = "$([char]0x1b)[93m"
    Red = "$([char]0x1b)[91m"
    Blue = "$([char]0x1b)[94m"
    Cyan = "$([char]0x1b)[96m"
    White = "$([char]0x1b)[97m"
    Bold = "$([char]0x1b)[1m"
    Reset = "$([char]0x1b)[0m"
}

# Function to print formatted messages
function Write-Header {
    param([string]$Message)
    Write-Host "`n$($Colors.Cyan)$($Colors.Bold)$('='*60)$($Colors.Reset)"
    Write-Host "$($Colors.Cyan)$($Colors.Bold)$($Message.PadLeft(30 + $Message.Length/2).PadRight(60))$($Colors.Reset)"
    Write-Host "$($Colors.Cyan)$($Colors.Bold)$('='*60)$($Colors.Reset)`n"
}

function Write-Success {
    param([string]$Message)
    Write-Host "$($Colors.Green)[OK] $Message$($Colors.Reset)"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "$($Colors.Yellow)[WARNING] $Message$($Colors.Reset)"
}

function Write-Error {
    param([string]$Message)
    Write-Host "$($Colors.Red)[ERROR] $Message$($Colors.Reset)"
}

function Write-Info {
    param([string]$Message)
    Write-Host "$($Colors.Blue)[INFO] $Message$($Colors.Reset)"
}

# Function to check if running as administrator
function Test-Admin {
    $currentUser = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to request admin privileges
function Request-AdminPrivileges {
    if (-not (Test-Admin)) {
        Write-Warning "Administrator privileges required for software installation."
        Write-Info "The script will attempt to restart with elevated privileges..."
        
        try {
            Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
            exit
        }
        catch {
            Write-Error "Failed to request admin privileges: $_"
            Write-Error "Please run the script as Administrator for automatic software installation."
            return $false
        }
    }
    return $true
}

# Function to download files
function Download-File {
    param(
        [string]$Url,
        [string]$FilePath,
        [string]$Description = ""
    )
    
    try {
        Write-Info "Downloading $($Description -or $FilePath)..."
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Url, $FilePath)
        Write-Success "Downloaded $FilePath"
        return $true
    }
    catch {
        Write-Error "Failed to download $FilePath`: $_"
        return $false
    }
}

# Function to check and install Git
function Install-GitIfMissing {
    Write-Info "Checking Git installation..."
    
    # Check if Git is installed
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $gitVersion = git --version
        Write-Success "Git is installed: $gitVersion"
        return $true
    }
    
    Write-Warning "Git is not installed. Installing Git for Windows..."
    
    # Download Git for Windows
    $gitUrl = "https://github.com/git-for-windows/git/releases/latest/download/Git-2.42.0.2-64-bit.exe"
    $gitInstaller = "Git-installer.exe"
    
    if (-not (Download-File -Url $gitUrl -FilePath $gitInstaller -Description "Git for Windows")) {
        return $false
    }
    
    # Install Git silently
    try {
        Write-Info "Installing Git for Windows..."
        Start-Process -FilePath ".\$gitInstaller" -ArgumentList "/SILENT /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh /TASKS=desktopicon" -Wait
        Write-Success "Git installed successfully"
        
        # Cleanup
        Remove-Item $gitInstaller -ErrorAction SilentlyContinue
        
        return $true
    }
    catch {
        Write-Error "Failed to install Git: $_"
        return $false
    }
}

# Function to check and install Azure CLI
function Install-AzureCliIfMissing {
    Write-Info "Checking Azure CLI installation..."
    
    # Check if Azure CLI is installed
    if (Get-Command az -ErrorAction SilentlyContinue) {
        Write-Success "Azure CLI is installed"
        return $true
    }
    
    Write-Warning "Azure CLI is not installed. Installing Azure CLI..."
    
    # Download Azure CLI installer
    $azureCliUrl = "https://aka.ms/installazurecliwindows"
    $azureCliInstaller = "AzureCLI.msi"
    
    if (-not (Download-File -Url $azureCliUrl -FilePath $azureCliInstaller -Description "Azure CLI")) {
        return $false
    }
    
    # Install Azure CLI silently
    try {
        Write-Info "Installing Azure CLI..."
        Start-Process msiexec -ArgumentList "/i `"$azureCliInstaller`" /quiet /norestart" -Wait
        Write-Success "Azure CLI installed successfully"
        
        # Cleanup
        Remove-Item $azureCliInstaller -ErrorAction SilentlyContinue
        
        return $true
    }
    catch {
        Write-Error "Failed to install Azure CLI: $_"
        return $false
    }
}

# Function to check and install VS Code
function Install-VSCodeIfMissing {
    Write-Info "Checking VS Code installation..."
    
    # Check if VS Code is installed
    if (Get-Command code -ErrorAction SilentlyContinue) {
        Write-Success "VS Code is installed and accessible via command line"
        return $true
    }
    
    Write-Warning "VS Code is not installed. Installing Visual Studio Code..."
    
    # Download VS Code installer
    $vsCodeUrl = "https://code.visualstudio.com/sha/download?build=stable&os=win32-x64"
    $vsCodeInstaller = "VSCodeSetup.exe"
    
    if (-not (Download-File -Url $vsCodeUrl -FilePath $vsCodeInstaller -Description "Visual Studio Code")) {
        return $false
    }
    
    # Install VS Code silently
    try {
        Write-Info "Installing Visual Studio Code..."
        Start-Process -FilePath ".\$vsCodeInstaller" -ArgumentList "/SILENT /mergetasks=!runcode,addcontextmenufiles,addcontextmenufolders,associatewithfiles,addtopath" -Wait
        Write-Success "VS Code installed successfully"
        
        # Install required extensions
        Write-Info "Installing VS Code extensions..."
        $extensions = @(
            "ms-python.python",
            "ms-vscode.powershell",
            "ms-vscode.vscode-json",
            "GitHub.copilot"
        )
        
        foreach ($extension in $extensions) {
            Write-Info "Installing $extension..."
            code --install-extension $extension
        }
        
        # Cleanup
        Remove-Item $vsCodeInstaller -ErrorAction SilentlyContinue
        
        return $true
    }
    catch {
        Write-Error "Failed to install VS Code: $_"
        return $false
    }
}

# Function to check and install AMO & ADOMD
function Install-AnalysisServicesIfMissing {
    Write-Info "Checking AMO & ADOMD libraries..."
    
    # Download Analysis Services client libraries
    $amoUrl = "https://go.microsoft.com/fwlink/?linkid=2180719"
    $adomdUrl = "https://go.microsoft.com/fwlink/?linkid=2180929"
    
    $amoInstaller = "AS_OLE_DB.msi"
    $adomdInstaller = "ADOMD_NET.msi"
    
    # Download both installers
    $amoSuccess = Download-File -Url $amoUrl -FilePath $amoInstaller -Description "Analysis Services OLE DB Provider"
    $adomdSuccess = Download-File -Url $adomdUrl -FilePath $adomdInstaller -Description "ADOMD.NET Provider"
    
    if (-not ($amoSuccess -and $adomdSuccess)) {
        return $false
    }
    
    # Install both packages
    try {
        Write-Info "Installing Analysis Services components..."
        Start-Process msiexec -ArgumentList "/i `"$amoInstaller`" /quiet /norestart" -Wait
        Start-Process msiexec -ArgumentList "/i `"$adomdInstaller`" /quiet /norestart" -Wait
        Write-Success "Analysis Services components installed successfully"
        
        # Cleanup
        Remove-Item $amoInstaller, $adomdInstaller -ErrorAction SilentlyContinue
        
        return $true
    }
    catch {
        Write-Error "Failed to install Analysis Services components: $_"
        return $false
    }
}

# Function to check and install .NET 8.0 SDK
function Install-DotNetSDKIfMissing {
    Write-Info "Checking .NET 8.0 SDK installation..."
    
    # Check if .NET SDK is installed
    if (Get-Command dotnet -ErrorAction SilentlyContinue) {
        $version = dotnet --version
        if ([version]$version -ge [version]"8.0.0") {
            Write-Success ".NET SDK version $version is installed"
            return $true
        }
    }
    
    Write-Warning ".NET 8.0 SDK not found. Installing..."
    
    # Download .NET 8.0 SDK installer
    $dotnetUrl = "https://dotnetcli.azureedge.net/dotnet/Sdk/8.0.403/dotnet-sdk-8.0.403-win-x64.exe"
    $installerPath = "dotnet-sdk-installer.exe"
    
    if (-not (Download-File -Url $dotnetUrl -FilePath $installerPath -Description ".NET 8.0 SDK")) {
        return $false
    }
    
    # Install .NET SDK silently
    try {
        Write-Info "Installing .NET 8.0 SDK (this may take several minutes)..."
        Start-Process -FilePath ".\$installerPath" -ArgumentList "/quiet /norestart" -Wait
        Write-Success ".NET 8.0 SDK installed successfully"
        
        # Cleanup
        Remove-Item $installerPath -ErrorAction SilentlyContinue
        
        return $true
    }
    catch {
        Write-Error "Failed to install .NET 8.0 SDK: $_"
        return $false
    }
}

# Main script execution
Write-Header "MCP Server Environment Setup"

# Check for admin privileges
if (-not (Request-AdminPrivileges)) {
    exit 1
}

# Install required software
$results = @{}

$results["git"] = Install-GitIfMissing
$results["azure_cli"] = Install-AzureCliIfMissing
$results["vscode"] = Install-VSCodeIfMissing
$results["dotnet"] = Install-DotNetSDKIfMissing

# Print summary
Write-Header "Installation Summary"
foreach ($item in $results.GetEnumerator()) {
    if ($item.Value) {
        Write-Success "$($item.Key): Installed successfully"
    }
    else {
        Write-Error "$($item.Key): Installation failed"
    }
}

Write-Host "`nSetup completed. Please restart your PowerShell session to ensure all changes take effect."