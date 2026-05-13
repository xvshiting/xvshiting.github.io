#!/bin/bash
# publish.sh - Build and deploy the site to GitHub Pages
# Usage: ./publish.sh [commit message]
#
# This script:
#   1. Commits all changes
#   2. Pushes to GitHub, which auto-deploys via GitHub Pages

set -e

MSG="${1:-Update site $(date +%Y-%m-%d)}"

echo "📦 Publishing site..."
echo ""

# Check for changes
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "✅ No changes to publish."
  exit 0
fi

# Add all changes
git add -A

# Commit
git commit -m "$MSG"
echo "✅ Committed: $MSG"

# Push
echo "🚀 Pushing to GitHub..."
git push origin main 2>/dev/null || git push origin master 2>/dev/null

echo ""
echo "✅ Done! Your site will be live in a few minutes at:"
echo "   https://xvshiting.github.io"