; Inno Setup Script for 4BiM Windows Installer
; Requires Inno Setup 6+ to compile

[Setup]
; App Information
AppName=4BiM Forensic Composite Generator
AppVersion=1.0.0
AppPublisher=Nathanael Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé
AppPublisherURL=https://github.com/votre-compte/ProjetBIM
AppSupportURL=https://github.com/votre-compte/ProjetBIM/issues
AppUpdatesURL=https://github.com/votre-compte/ProjetBIM/releases

; Default Installation Paths
DefaultDirName={autopf}\4BiM
DefaultGroupName=4BiM

; Output Configuration
OutputDir=dist\Installers
OutputBaseFilename=4BiM_Setup_Windows

; Theming and Assets
SetupIconFile=programme\src\assets\icon.ico
UninstallDisplayIcon={app}\4BiM.exe
Compression=lzma2/ultra64
SolidCompression=yes

; User Experience
LicenseFile=LICENSE
WizardStyle=modern

[Files]
; Copy all generated PyInstaller core files
Source: "dist\4BiM\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Copy any external data if they weren't fully bundled inside the exe (ensure paths match your built directory)
; Example: Source: "data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu Shortcut
Name: "{group}\4BiM"; Filename: "{app}\4BiM.exe"; IconFilename: "{app}\4BiM.exe"
Name: "{group}\Uninstall 4BiM"; Filename: "{uninstallexe}"
; Desktop Shortcut
Name: "{autodesktop}\4BiM"; Filename: "{app}\4BiM.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
; Option to launch right after installing
Filename: "{app}\4BiM.exe"; Description: "{cm:LaunchProgram,4BiM}"; Flags: nowait postinstall skipifsilent
