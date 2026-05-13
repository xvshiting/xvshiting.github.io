# Xu Shiting's Personal Homepage

A Jekyll-powered personal academic homepage hosted on GitHub Pages.

## 🚀 Quick Start

### Create a New Blog Post

```bash
./newpost.sh "My Article Title"
./newpost.sh "Deep Learning Notes" "Deep Learning,NLP"
```

This creates a new markdown file in `_posts/` with proper frontmatter. Edit it with your content.

### Update Bio Info

```bash
./update_bio.sh --show                  # View current bio
./update_bio.sh --position "Professor"  # Update position
./update_bio.sh --news "Published new paper!"  # Add news item
```

### Publish Changes

```bash
./publish.sh "Add new post about NLP"
```

This commits all changes and pushes to GitHub, which auto-deploys via GitHub Pages.

## 📁 Site Structure

```
├── _config.yml          # Site config (bio, nav, etc.)
├── _layouts/            # HTML templates
│   ├── default.html     # Main layout (sidebar + content)
│   ├── post.html        # Blog post layout
│   └── author.html      # Author page layout
├── _includes/           # Reusable HTML components
│   └── navigation.html  # Sidebar navigation
├── _posts/              # Blog posts (Markdown)
│   └── YYYY-MM-DD-title.md
├── _sass/               # SCSS stylesheets
│   └── main.scss
├── assets/              # Static files
│   ├── css/
│   ├── image/
│   └── pdf/
├── about.md             # About page
├── blog.html            # Blog listing page
├── index.md             # Homepage
├── mytags.html          # Tags page
├── newpost.sh           # Script: create new post
├── update_bio.sh        # Script: update bio info
└── publish.sh           # Script: commit & push to GitHub
```

## ✍️ Writing a Post

1. Run `./newpost.sh "Your Title" "Tag1,Tag2"`
2. Edit the created file in `_posts/`
3. Add your content in Markdown
4. Run `./publish.sh "Add post about X"`

## 🎨 Customizing

- **Bio & Links**: Edit `author` section in `_config.yml`
- **Navigation**: Edit `nav` section in `_config.yml`
- **Colors**: Edit CSS variables at top of `_sass/main.scss`
- **Homepage sections**: Edit `index.md`
- **About page**: Edit `about.md`

## 🏗️ Local Development

```bash
bundle install
bundle exec jekyll serve
# Open http://localhost:4000
```