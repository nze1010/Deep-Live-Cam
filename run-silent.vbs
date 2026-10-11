Set WshShell = CreateObject("WScript.Shell")
strDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
WshShell.CurrentDirectory = strDir
WshShell.Run """" & strDir & "\venv\Scripts\pythonw.exe"" run.py --execution-provider dml", 0, False
