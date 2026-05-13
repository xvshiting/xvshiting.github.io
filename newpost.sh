#!/bin/bash
# newpost.sh - Create a new blog post with proper frontmatter
# Usage: ./newpost.sh "Post Title" [tags] [series_name] [series_order]
#
# Examples:
#   ./newpost.sh "My New Article"
#   ./newpost.sh "Deep Learning Notes" "Deep Learning,NLP"
#   ./newpost.sh "Agent Ch 1: Intro" "Agent LLM" "Building AI Agents" 1
#   ./newpost.sh "Agent Ch 2: Tools" "Agent LLM" "Building AI Agents" 2

set -e

POSTS_DIR="_posts"
AUTHOR="willXu"

if [ -z "$1" ]; then
  echo "❌ Usage: ./newpost.sh \"Post Title\" [tags] [series_name] [series_order]"
  echo ""
  echo "Examples:"
  echo "  ./newpost.sh \"My New Article\""
  echo "  ./newpost.sh \"Deep Learning Notes\" \"Deep Learning,NLP\""
  echo "  ./newpost.sh \"Agent Ch 1\" \"Agent LLM\" \"Building AI Agents\" 1"
  exit 1
fi

TITLE="$1"
TAGS="${2:-}"
SERIES="${3:-}"
ORDER="${4:-}"

# Convert title to slug for filename
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
DATE=$(date +%Y-%m-%d)
FILENAME="${POSTS_DIR}/${DATE}-${SLUG}.md"

if [ -f "$FILENAME" ]; then
  echo "❌ File already exists: $FILENAME"
  exit 1
fi

# Build tag frontmatter
if [ -n "$TAGS" ]; then
  TAG_LINE="tag: [${TAGS}]"
else
  TAG_LINE="# tag: [tag1, tag2]"
fi

# Build series frontmatter
SERIES_LINES=""
if [ -n "$SERIES" ]; then
  SERIES_LINES="series: \"${SERIES}\""
  if [ -n "$ORDER" ]; then
    SERIES_LINES="${SERIES_LINES}
series_order: ${ORDER}"
  fi
fi

cat > "$FILENAME" <<EOF
---
layout: post
author: ${AUTHOR}
${TAG_LINE}
${SERIES_LINES}
---

# ${TITLE}

Write your content here. Markdown is supported!

EOF

echo "✅ Created: $FILENAME"
echo "📝 Open it and start writing!"
if [ -n "$SERIES" ]; then
  echo "📚 Series: ${SERIES}"$([ -n "$ORDER" ] && echo " (Chapter ${ORDER})")
fi
echo ""
echo "💡 Don't forget to:"
echo "   - Add tags to the frontmatter"
echo "   - Write an excerpt at the top (first paragraph becomes excerpt)"
echo "   - Run ./publish.sh when ready to publish"