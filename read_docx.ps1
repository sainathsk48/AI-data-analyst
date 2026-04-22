Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead('C:\Users\saina\Downloads\New folder\AI_DataAnalyst_Spec.docx')
$entry = $zip.Entries | Where-Object { $_.FullName -eq 'word/document.xml' }
$stream = $entry.Open()
$reader = New-Object System.IO.StreamReader($stream)
$xml = [xml]$reader.ReadToEnd()
$reader.Close()
$zip.Dispose()
$xml.SelectNodes('//*[local-name()="t"]') | ForEach-Object { $_.InnerText }
