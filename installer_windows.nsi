!define APP_NAME "4BiM"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "Nathanael Pons et al."

Name "${APP_NAME}"
OutFile "dist/Installers/4BiM_Setup_Windows.exe"
InstallDir "$PROFILE\${APP_NAME}"

RequestExecutionLevel user

Page directory
Page instfiles

Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    
    ; Copy source files directly (No compilation!)
    SetOutPath "$INSTDIR\programme"
    File /r /x "venv" /x "__pycache__" "programme\*"
    SetOutPath "$INSTDIR"
    File /r "data"
    File "run.bat"
    File "LICENSE"
    File "README.md"
    
    ; Create a shortcut on the Desktop that executes run.bat
    CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\run.bat" "" "$INSTDIR\programme\src\assets\icon.ico"
    
    ; Create Start Menu shortcuts
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\run.bat" "" "$INSTDIR\programme\src\assets\icon.ico"
    CreateShortCut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
    
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
    RMDir /r "$INSTDIR"
    Delete "$DESKTOP\${APP_NAME}.lnk"
    RMDir /r "$SMPROGRAMS\${APP_NAME}"
SectionEnd
