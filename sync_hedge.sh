#!/bin/bash
# Sync latest Hedge*.xlsx from OneDrive to local data dir
# Runs via launchd every 5 minutes
# Uses Finder via osascript to bypass macOS TCC restrictions on ~/Library/CloudStorage

DST="$HOME/geomet-copper-dashboard/data"
mkdir -p "$DST"

result=$(osascript -e "
tell application \"Finder\"
    set srcFolder to folder \"Pricing_Hedge - Hedge Worksheet\" of folder \"OneDrive-GeometRecycle\" of folder \"CloudStorage\" of folder \"Library\" of (path to home folder)
    set dstFolder to POSIX file \"$DST\" as alias
    set allFiles to (every file of srcFolder whose name begins with \"Hedge\" and name extension is \"xlsx\")
    if (count of allFiles) = 0 then return \"none\"
    set latest to item 1 of allFiles
    repeat with f in allFiles
        if name of f > name of latest then set latest to f
    end repeat
    set latestName to name of latest
    set dstFile to (POSIX path of (dstFolder as alias)) & latestName
    try
        POSIX file dstFile as alias
        set dstMod to modification date of (POSIX file dstFile as alias)
        set srcMod to modification date of latest
        if srcMod > dstMod then
            duplicate latest to dstFolder with replacing
            return \"copied \" & latestName
        else
            return \"current \" & latestName
        end if
    on error
        duplicate latest to dstFolder with replacing
        return \"copied \" & latestName
    end try
end tell
" 2>/dev/null)

case "$result" in
    copied*)  echo "[sync_hedge] Copied ${result#copied }" ;;
    current*) echo "[sync_hedge] ${result#current } is up to date" ;;
    none)     echo "[sync_hedge] No Hedge*.xlsx files in OneDrive" ;;
    *)        echo "[sync_hedge] Unexpected: $result" ;;
esac
