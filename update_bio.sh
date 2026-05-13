#!/bin/bash
# update_bio.sh - Update bio information in _config.yml
# Usage: ./update_bio.sh [option] [value]
#
# Options:
#   --name "Your Name"           Update full name
#   --name-en "English Name"      Update English name
#   --email "email@example.com"   Update email
#   --position "Position"         Update position/title
#   --department "Department"     Update department
#   --university "University"     Update university
#   --news "News text"            Add a news item to homepage
#   --show                        Show current bio info
#
# Examples:
#   ./update_bio.sh --show
#   ./update_bio.sh --position "Associate Professor"
#   ./update_bio.sh --news "Got promoted to Associate Professor!"

CONFIG="_config.yml"
INDEX="index.md"

show_bio() {
  echo "📋 Current Bio Information:"
  echo "========================"
  grep -A 20 "^author:" "$CONFIG" | head -20
  echo ""
  echo "📰 Current News:"
  grep -A 5 "News" "$INDEX" | head -6
}

add_news() {
  local news_text="$1"
  local news_date=$(date +%Y-%m-%d)
  
  # Add news item after the news section opening
  if grep -q "<!-- NEWS_ITEMS -->" "$INDEX"; then
    sed -i.bak "/<!-- NEWS_ITEMS -->/a\\
      <li><span class=\"news-date\">${news_date}</span> ${news_text}</li>" "$INDEX"
    rm -f "${INDEX}.bak"
  else
    # Fallback: insert after the news-list ul opening
    sed -i.bak "/<ul class=\"news-list\">/a\\
      <li><span class=\"news-date\">${news_date}</span> ${news_text}</li>" "$INDEX"
    rm -f "${INDEX}.bak"
  fi
  echo "✅ News added: [$news_date] $news_text"
}

update_config_field() {
  local field="$1"
  local value="$2"
  
  if grep -q "^  ${field}:" "$CONFIG"; then
    sed -i.bak "s/^  ${field}:.*/  ${field}: \"${value}\"/" "$CONFIG"
    rm -f "${CONFIG}.bak"
    echo "✅ Updated ${field} to: ${value}"
  else
    echo "❌ Field '${field}' not found in ${CONFIG}"
  fi
}

# Main
case "$1" in
  --show)
    show_bio
    ;;
  --name)
    [ -z "$2" ] && echo "❌ Please provide a name" && exit 1
    update_config_field "name" "$2"
    ;;
  --name-en)
    [ -z "$2" ] && echo "❌ Please provide an English name" && exit 1
    update_config_field "name_en" "$2"
    ;;
  --email)
    [ -z "$2" ] && echo "❌ Please provide an email" && exit 1
    update_config_field "email" "$2"
    ;;
  --position)
    [ -z "$2" ] && echo "❌ Please provide a position" && exit 1
    update_config_field "position" "$2"
    ;;
  --department)
    [ -z "$2" ] && echo "❌ Please provide a department" && exit 1
    update_config_field "department" "$2"
    ;;
  --university)
    [ -z "$2" ] && echo "❌ Please provide a university" && exit 1
    update_config_field "university" "$2"
    ;;
  --news)
    [ -z "$2" ] && echo "❌ Please provide news text" && exit 1
    add_news "$2"
    ;;
  *)
    echo "📖 update_bio.sh - Update bio information"
    echo ""
    echo "Usage: ./update_bio.sh [option] [value]"
    echo ""
    echo "Options:"
    echo "  --name \"Your Name\"           Update full name"
    echo "  --name-en \"English Name\"     Update English name"
    echo "  --email \"email@example.com\"   Update email"
    echo "  --position \"Position\"         Update position/title"
    echo "  --department \"Department\"     Update department"
    echo "  --university \"University\"     Update university"
    echo "  --news \"News text\"            Add a news item"
    echo "  --show                        Show current bio"
    ;;
esac