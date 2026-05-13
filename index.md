---
layout: default
title: Home
---
<div class="home">
  <!-- Bio Section -->
  <section class="section bio-section">
    <p class="bio-quote"><em>"NLP as well as Deep Learning could not only help machines better understand human beings, but also help us to know ourselves better."</em></p>
    <p>
      My name is <strong>S. Xu (许士亭)</strong>, also known as <strong>Will Xu</strong>.
      I am a lecturer at the <strong>Department of Cyberspace Security</strong> in <strong>Shandong University of Political Science and Law</strong>.
    </p>
    <p>
      My research interests are <em>Machine Learning</em>, <em>NLP</em>, <em>Deep Learning</em> and <em>Data Mining</em>.
      I am very interested in erecting novel models and admiring effects they bring to the world.
      I have also worked on <em>Information Security related to malicious software classification</em>.
    </p>
    <div class="bio-links">
      <a href="mailto:{{ site.author.email }}" class="btn"><i class="fa fa-envelope"></i> Contact</a>
      <a href="{{ site.author.cv }}" class="btn"><i class="fa fa-file-pdf-o"></i> CV</a>
      <a href="https://github.com/{{ site.author.github }}" class="btn"><i class="fa fa-github"></i> GitHub</a>
    </div>
  </section>

  <!-- News Section -->
  <section class="section">
    <h2 class="section-title"><i class="fa fa-bullhorn"></i> News</h2>
    <ul class="news-list">
      <!-- NEWS_ITEMS -->
      <li><span class="news-date">2022-09-01</span> I am a lecturer at Department of Cyberspace Security in Shandong University of Political Science and Law.</li>
    </ul>
  </section>

  <!-- Publications Section -->
  <section class="section">
    <h2 class="section-title"><i class="fa fa-file-text-o"></i> Publications</h2>

    <h3 class="pub-subtitle">Journal Papers</h3>
    <ol class="publication-list">
      <li>
        <span class="pub-year">2026</span>
        <strong>S. Xu</strong>
        <a href="https://link.springer.com/article/10.1186/s42400-025-00481-3">VIMAR: vision-language informed malware analysis and reasoning model.</a>
        <em>Cybersecurity</em>, 9, 49, 2026.
        <span class="pub-type journal">journal</span>
      </li>
      <li>
        <span class="pub-year">2025</span>
        <strong>S. Xu</strong>
        <a href="https://doi.org/10.1016/j.ins.2025.122470">DEEP-CWS: Distilling Efficient pre-trained models with Early exit and Pruning for scalable Chinese Word Segmentation.</a>
        <em>Information Sciences</em>, 719, 122470, 2025.
        <span class="pub-type journal">journal</span>
      </li>
    </ol>

    <h3 class="pub-subtitle">Conference Papers</h3>
    <ol class="publication-list">
      <li>
        <span class="pub-year">2024</span>
        <strong>S. Xu.</strong>
        <a href="https://doi.org/10.1145/3654823.3654872">BED: Chinese Word Segmentation Model Based on Boundary-Enhanced Decoder.</a>
        <em>CACML</em>, 2024.
        <span class="pub-type conference">conference</span>
      </li>
      <li>
        <span class="pub-year">2021</span>
        <strong>S. Xu,</strong> et al.
        <a href="https://link.springer.com/chapter/10.1007/978-3-030-78292-4_36">Automatic Task Requirements Writing Evaluation via Machine Reading Comprehension.</a>
        <em>AIED</em>, 2021.
        <a href="/assets/pdf/paper/202101.pdf" class="pub-pdf">[pdf]</a>
        <span class="pub-type conference">conference</span>
      </li>
      <li>
        <span class="pub-year">2020</span>
        <strong>S. Xu,</strong> W. Ding, and Z. Liu.
        <a href="https://link.springer.com/chapter/10.1007/978-3-030-52240-7_62">Automatic Dialogic Instruction Detection for K-12 Online One-on-One Classes.</a>
        <em>AIED</em>, 2020.
        <a href="/assets/pdf/paper/202001.pdf" class="pub-pdf">[pdf]</a>
        <span class="pub-type conference">conference</span>
      </li>
      <li>
        <span class="pub-year">2017</span>
        <strong>S. Xu,</strong> X. Ma, Y. Liu, and Q. Sheng.
        <a href="http://ieeexplore.ieee.org/document/7917194/">Malicious Application Dynamic Detection in Real-Time API Analysis.</a>
        <em>IEEE iThings-GreenCom-CPSCom-SmartData</em>, pp. 788-794, 2016.
        <a href="/assets/pdf/paper/5880a788.pdf" class="pub-pdf">[pdf]</a>
        <span class="pub-type conference">conference</span>
      </li>
    </ol>
  </section>

  <!-- Projects Section -->
  <section class="section">
    <h2 class="section-title"><i class="fa fa-code"></i> Projects</h2>
    <div class="card-grid">
      {% for project in site.projects %}
      <div class="card{% if project.highlight %} card-highlight{% endif %}">
        <div class="card-header">
          <h3 class="card-title">{{ project.name }}</h3>
          {% if project.highlight %}
          <span class="card-badge"><i class="fa fa-star"></i></span>
          {% endif %}
        </div>
        <p class="card-desc">{{ project.description }}</p>
        <div class="card-tags">
          {% for t in project.tags %}
          <span class="tag">{{ t }}</span>
          {% endfor %}
        </div>
        <div class="card-links">
          {% if project.url %}
          <a href="{{ project.url }}" class="btn"><i class="fa fa-link"></i> View</a>
          {% endif %}
          {% if project.github %}
          <a href="https://github.com/{{ project.github }}" class="btn btn-outline"><i class="fa fa-github"></i> Source</a>
          {% endif %}
        </div>
      </div>
      {% endfor %}
    </div>
    <a href="/projects.html" class="see-all">View all projects →</a>
  </section>

  <!-- Experience Section -->
  <section class="section">
    <h2 class="section-title"><i class="fa fa-briefcase"></i> Experience</h2>
    <div class="timeline">
      <div class="timeline-item">
        <div class="timeline-date">2021.09 – 2022.07</div>
        <div class="timeline-content">
          <h3>Du Xiao Man <span class="timeline-location">Beijing, China</span></h3>
          <p>Responsible for NLP infrastructure. Focused on Chinese Word Segmentation Task.</p>
        </div>
      </div>
      <div class="timeline-item">
        <div class="timeline-date">2019.08 – 2021.09</div>
        <div class="timeline-content">
          <h3>Tomorrow Advanced Life <span class="timeline-location">Beijing, China</span></h3>
          <p>Working on Chinese writing judgement system. Developed Chinese Word Correction model based on pre-trained language model.</p>
          <p>Working on English writing evaluation, responsible for the whole system. Focused on English Grammar Correction task based on Transformer architecture. Also built a prompt writing task evaluation model based on MRC technology.</p>
        </div>
      </div>
      <div class="timeline-item">
        <div class="timeline-date">2017.09 – 2018.11</div>
        <div class="timeline-content">
          <h3>Pachira Information Technology <span class="timeline-location">Beijing, China</span></h3>
          <p>Improved Role accuracy of speech translation model with seq2seq model based on semantic information.</p>
          <p>Participated in building a system based on Question-Answer model to extract user information from conversations.</p>
        </div>
      </div>
      <div class="timeline-item">
        <div class="timeline-date">2017.03 – 2017.07</div>
        <div class="timeline-content">
          <h3>Kaspersky Lab <span class="timeline-location">Beijing, China</span> <span class="tag">Internship</span></h3>
          <p>Designed a malicious software family classification model based on CNN. <a href="/2019/05/19/research-summary.html">[Details]</a></p>
          <p>Implemented a CS system (based on tornado) to help analysts train and invoke the model.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Education Section -->
  <section class="section">
    <h2 class="section-title"><i class="fa fa-graduation-cap"></i> Education</h2>
    <div class="timeline">
      <div class="timeline-item">
        <div class="timeline-date">2014.09 – 2017.03</div>
        <div class="timeline-content">
          <h3>Master of Science</h3>
          <p>School of Cyberspace Security (Former School of Computer Science), Beijing University of Posts and Telecommunications</p>
        </div>
      </div>
      <div class="timeline-item">
        <div class="timeline-date">2010.09 – 2014.07</div>
        <div class="timeline-content">
          <h3>Bachelor of Engineering</h3>
          <p>Computer Science Department, Shandong University of Technology</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Awards Section -->
  <section class="section">
    <h2 class="section-title"><i class="fa fa-trophy"></i> Awards</h2>
    <ul class="simple-list">
      <li>2014.9 – 2017.3 &ensp; The First Honor Graduate Scholarship for 3 consecutive years</li>
    </ul>
  </section>

  <!-- Blog Section -->
  <section class="section">
    <h2 class="section-title"><i class="fa fa-pencil"></i> Recent Posts</h2>
    <ul class="post-list">
      {% for post in site.posts limit:5 %}
      <li>
        <span class="post-date">{{ post.date | date_to_string }}</span>
        <a href="{{ post.url }}">{{ post.title }}</a>
        {% for t in post.tag %}
        <a href="/mytags.html#{{ t | slugize }}" class="tag">{{ t }}</a>
        {% endfor %}
      </li>
      {% endfor %}
    </ul>
    <a href="/blog.html" class="see-all">View all posts →</a>
  </section>
</div>