#!/bin/bash
# newpost.sh - Create a new blog post with proper frontmatter
# Usage: ./newpost.sh "Post Title" [category1,category2,...]
#
# Examples:
#   ./newpost.sh "My New Article"
#   ./newpost.sh "Deep Learning Notes" "Deep Learning,NLP"
#   ./newpost.sh "Malware Analysis Part 2" "Information Security,Deep Learning"

set -e

POSTS_DIR="_posts"
AUTHOR="willXu"

if [ -z "$1" ]; then
  echo "❌ Usage: ./newpost.sh \"Post Title\" [tag1,tag2,...]"
  echo ""
  echo "Examples:"
  echo "  ./newpost.sh \"My New Article\""
  echo "  ./newpost.sh \"Deep Learning Notes\" \"Deep Learning,NLP\""
  exit 1
fi

TITLE="$1"
TAGS="${2:-}"

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

cat > "$FILENAME" <<EOF
---
layout: post
author: ${AUTHOR}
${TAG_LINE}
---

# ${TITLE}

Write your content here. Markdown is supported!

EOF

echo "✅ Created: $FILENAME"
echo "📝 Open it and start writing!"
echo ""
echo "💡 Don't forget to:"
echo "   - Add tags to the frontmatter"
echo "   - Write an excerpt at the top (first paragraph becomes excerpt)"
echo "   - Run ./publish.sh when ready to publish"