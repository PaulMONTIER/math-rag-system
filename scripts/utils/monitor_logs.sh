#!/bin/bash
echo "🔍 Monitoring des logs RAG système..."
echo "Appuyez sur Ctrl+C pour arrêter"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

tail -f "/Users/paul/Desktop/Cours M2 /Projet Math/math-rag-system/data/logs/app.log" | grep --line-buffered -E "(embed|retrieval|generation|classifier|GPT|Claude)" --color=always
