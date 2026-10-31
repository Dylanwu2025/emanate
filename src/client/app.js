const AGENT_META = {
  maya: {
    name: 'Maya',
    initial: 'M',
    role: 'Flights & Budget',
  },
  liam: {
    name: 'Liam',
    initial: 'L',
    role: 'Transit & Food',
  },
  zoe: {
    name: 'Zoe',
    initial: 'Z',
    role: 'Group & Dietary',
  },
};

const MENTION_CANDIDATES = [
  { id: 'maya', tag: '@Maya', name: 'Maya', role: 'Flights, Hotels & Budget' },
  { id: 'liam', tag: '@Liam', name: 'Liam', role: 'Local Guide — Transit & Food' },
  { id: 'zoe', tag: '@Zoe', name: 'Zoe', role: 'Group Pacing & Dietary' },
];

const messagesById = new Map();

const feedEl = document.getElementById('message-feed');
const listEl = document.getElementById('messages-list');
const emptyStateEl = document.getElementById('empty-state');
const formEl = document.getElementById('composer-form');
const inputEl = document.getElementById('composer-input');
const stopBtnEl = document.getElementById('stop-btn');
const resetBtnEl = document.getElementById('reset-btn');
const mentionMenuEl = document.getElementById('mention-menu');

let isPinnedToBottom = true;

feedEl.addEventListener('scroll', () => {
  const distanceFromBottom =
    feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight;
  isPinnedToBottom = distanceFromBottom < 72;
});

function scrollToBottomIfPinned(force = false) {
  if (force || isPinnedToBottom) {
    feedEl.scrollTop = feedEl.scrollHeight;
  }
}

function hasActiveStream() {
  for (const msg of messagesById.values()) {
    if (msg.status === 'streaming') {
      return true;
    }
  }
  return false;
}

function syncToolbarState() {
  if (hasActiveStream()) {
    stopBtnEl.classList.remove('hidden');
  } else {
    stopBtnEl.classList.add('hidden');
  }

  if (messagesById.size === 0) {
    emptyStateEl.classList.remove('hidden');
  } else {
    emptyStateEl.classList.add('hidden');
  }
}

function getSortedMessages() {
  return Array.from(messagesById.values()).sort((a, b) => a.seq - b.seq);
}

function createMessageElement(msg) {
  const rowEl = document.createElement('div');
  rowEl.className = `message-row role-${msg.role}${msg.agentId ? ` agent-${msg.agentId}` : ''}`;
  rowEl.dataset.messageId = msg.id;
  rowEl.dataset.seq = String(msg.seq);

  if (msg.role === 'agent') {
    const meta = AGENT_META[msg.agentId] || {
      name: msg.agentName || 'Agent',
      initial: (msg.agentName || 'A')[0],
      role: '',
    };

    const avatarEl = document.createElement('div');
    avatarEl.className = `avatar avatar-${msg.agentId || 'maya'} message-avatar`;
    avatarEl.textContent = meta.initial;
    rowEl.appendChild(avatarEl);

    const wrapEl = document.createElement('div');
    wrapEl.className = 'message-bubble-wrap';

    const senderLine = document.createElement('div');
    senderLine.className = 'message-sender-line';

    const nameEl = document.createElement('span');
    nameEl.className = `sender-name sender-name-${msg.agentId || 'maya'}`;
    nameEl.textContent = meta.name;
    senderLine.appendChild(nameEl);

    if (meta.role) {
      const roleEl = document.createElement('span');
      roleEl.className = 'sender-role-tag';
      roleEl.textContent = meta.role;
      senderLine.appendChild(roleEl);
    }

    wrapEl.appendChild(senderLine);

    const bubbleEl = document.createElement('div');
    bubbleEl.className = 'message-bubble';
    updateBubbleContent(bubbleEl, msg);
    wrapEl.appendChild(bubbleEl);

    rowEl.appendChild(wrapEl);
  } else {
    const wrapEl = document.createElement('div');
    wrapEl.className = 'message-bubble-wrap';

    const bubbleEl = document.createElement('div');
    bubbleEl.className = 'message-bubble';
    updateBubbleContent(bubbleEl, msg);
    wrapEl.appendChild(bubbleEl);

    rowEl.appendChild(wrapEl);
  }

  return rowEl;
}

function escapeHtml(raw) {
  return String(raw)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderInlineMarkdown(escapedLine) {
  return escapedLine
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
}

function renderMarkdownSafe(rawText) {
  if (!rawText) return '';
  const escaped = escapeHtml(rawText);
  const lines = escaped.split(/\r?\n/);

  const htmlParts = [];
  let inUl = false;
  let inOl = false;
  let paragraphLines = [];

  const flushParagraph = () => {
    if (paragraphLines.length > 0) {
      htmlParts.push(`<p>${paragraphLines.join('<br/>')}</p>`);
      paragraphLines = [];
    }
  };

  const closeLists = () => {
    if (inUl) {
      htmlParts.push('</ul>');
      inUl = false;
    }
    if (inOl) {
      htmlParts.push('</ol>');
      inOl = false;
    }
  };

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) {
      flushParagraph();
      closeLists();
      continue;
    }

    const headingMatch = rawLine.match(/^\s*(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      closeLists();
      const level = Math.min(headingMatch[1].length + 2, 5);
      htmlParts.push(`<h${level} class="md-heading">${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      continue;
    }

    const ulMatch = rawLine.match(/^\s*[-*•]\s+(.*)$/);
    if (ulMatch) {
      flushParagraph();
      if (inOl) {
        htmlParts.push('</ol>');
        inOl = false;
      }
      if (!inUl) {
        htmlParts.push('<ul>');
        inUl = true;
      }
      htmlParts.push(`<li>${renderInlineMarkdown(ulMatch[1])}</li>`);
      continue;
    }

    const olMatch = rawLine.match(/^\s*\d+[.)]\s+(.*)$/);
    if (olMatch) {
      flushParagraph();
      if (inUl) {
        htmlParts.push('</ul>');
        inUl = false;
      }
      if (!inOl) {
        htmlParts.push('<ol>');
        inOl = true;
      }
      htmlParts.push(`<li>${renderInlineMarkdown(olMatch[1])}</li>`);
      continue;
    }

    closeLists();
    paragraphLines.push(renderInlineMarkdown(rawLine));
  }

  flushParagraph();
  closeLists();
  return htmlParts.join('');
}

function getOrCreateTypingDots(bubbleEl) {
  if (!bubbleEl._typingDotsEl) {
    const dots = document.createElement('span');
    dots.className = 'typing-dots';
    dots.setAttribute('aria-hidden', 'true');
    dots.innerHTML = '<span></span><span></span><span></span>';
    bubbleEl._typingDotsEl = dots;
  }
  return bubbleEl._typingDotsEl;
}

function updateBubbleContent(bubbleEl, msg) {
  const dots = bubbleEl._typingDotsEl;
  if (dots && dots.parentNode) {
    dots.parentNode.removeChild(dots);
  }

  const nextContent = msg.content || '';
  if (bubbleEl._renderedContent !== nextContent) {
    bubbleEl.innerHTML = renderMarkdownSafe(nextContent);
    bubbleEl._renderedContent = nextContent;
  }

  if (msg.status === 'streaming') {
    const typingDots = getOrCreateTypingDots(bubbleEl);
    const lastBlock = bubbleEl.lastElementChild;
    if (lastBlock && (lastBlock.tagName === 'P' || lastBlock.tagName === 'LI')) {
      lastBlock.appendChild(typingDots);
    } else {
      bubbleEl.appendChild(typingDots);
    }
  }
}

function renderAllMessages(forceScroll = false) {
  const sorted = getSortedMessages();
  listEl.innerHTML = '';
  for (const msg of sorted) {
    listEl.appendChild(createMessageElement(msg));
  }
  syncToolbarState();
  scrollToBottomIfPinned(forceScroll);
}

function upsertMessageInDOM(msg, forceScroll = false) {
  messagesById.set(msg.id, msg);
  const existingEl = listEl.querySelector(`[data-message-id="${CSS.escape(msg.id)}"]`);

  if (existingEl) {
    const bubbleEl = existingEl.querySelector('.message-bubble');
    if (bubbleEl) {
      updateBubbleContent(bubbleEl, msg);
    }
    syncToolbarState();
    scrollToBottomIfPinned(forceScroll);
    return;
  }

  const newEl = createMessageElement(msg);
  const children = Array.from(listEl.children);
  const insertBeforeEl = children.find(
    (child) => Number(child.dataset.seq || 0) > msg.seq,
  );

  if (insertBeforeEl) {
    listEl.insertBefore(newEl, insertBeforeEl);
  } else {
    listEl.appendChild(newEl);
  }

  syncToolbarState();
  scrollToBottomIfPinned(forceScroll);
}

async function fetchInitialState() {
  const res = await fetch('/api/state');
  if (!res.ok) return;
  const data = await res.json();
  messagesById.clear();
  for (const msg of data.messages || []) {
    messagesById.set(msg.id, msg);
  }
  renderAllMessages(true);
}

let activeEventSource = null;
let sseReconnectTimer = null;

function connectSSE() {
  if (sseReconnectTimer) {
    clearTimeout(sseReconnectTimer);
    sseReconnectTimer = null;
  }
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }

  const es = new EventSource('/api/stream');
  activeEventSource = es;

  es.onopen = () => {
    fetchInitialState().catch(() => {});
  };

  es.onmessage = (ev) => {
    try {
      const event = JSON.parse(ev.data);
      if (event.type === 'message.created' && event.message) {
        upsertMessageInDOM(event.message, event.message.role === 'user');
      } else if (event.type === 'message.delta') {
        const existing = messagesById.get(event.id);
        if (existing) {
          existing.content = (existing.content || '') + event.delta;
          existing.status = 'streaming';
          upsertMessageInDOM(existing, false);
        }
      } else if (event.type === 'message.done' && event.message) {
        upsertMessageInDOM(event.message, false);
      } else if (event.type === 'turn.idle') {
        for (const msg of messagesById.values()) {
          if (msg.status === 'streaming') {
            msg.status = 'completed';
            upsertMessageInDOM(msg, false);
          }
        }
        syncToolbarState();
      }
    } catch {
      // Ignore malformed frame
    }
  };

  es.onerror = () => {
    setTimeout(() => {
      fetchInitialState().catch(() => {});
    }, 1000);

    if (es.readyState === EventSource.CLOSED) {
      if (activeEventSource === es) {
        activeEventSource = null;
      }
      if (!sseReconnectTimer) {
        sseReconnectTimer = setTimeout(() => {
          sseReconnectTimer = null;
          connectSSE();
        }, 1500);
      }
    }
  };
}

async function sendUserMessage(rawText) {
  const content = rawText.trim();
  if (!content) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  hideMentionMenu();
  inputEl.focus();

  if (!activeEventSource || activeEventSource.readyState === EventSource.CLOSED) {
    connectSSE();
  }

  try {
    const res = await fetch('/api/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
    if (!res.ok) {
      if (!inputEl.value) {
        inputEl.value = rawText;
      }
      return;
    }
    const data = await res.json().catch(() => null);
    if (data) {
      if (data.userMessage) {
        upsertMessageInDOM(data.userMessage, true);
      }
      if (Array.isArray(data.messages)) {
        for (const msg of data.messages) {
          upsertMessageInDOM(msg, false);
        }
      }
    }
  } catch {
    if (!inputEl.value) {
      inputEl.value = rawText;
    }
  }
}

formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  sendUserMessage(inputEl.value);
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendUserMessage(inputEl.value);
  } else if (e.key === 'Escape') {
    hideMentionMenu();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 140)}px`;
  checkMentionAutocomplete();
});

function insertMentionIntoComposer(mentionText) {
  const start = inputEl.selectionStart ?? inputEl.value.length;
  const end = inputEl.selectionEnd ?? inputEl.value.length;
  const before = inputEl.value.slice(0, start);
  const after = inputEl.value.slice(end);

  const atMatch = before.match(/(?:^|\s)@([a-zA-Z]*)$/);
  let prefix;
  if (atMatch) {
    const replaceStart = before.lastIndexOf('@');
    prefix = before.slice(0, replaceStart) + mentionText;
  } else {
    const needsSpace = before.length > 0 && !/\s$/.test(before);
    prefix = before + (needsSpace ? ' ' : '') + mentionText;
  }

  inputEl.value = prefix + after;
  hideMentionMenu();
  inputEl.focus();
  const cursorPos = prefix.length;
  inputEl.setSelectionRange(cursorPos, cursorPos);
}

function checkMentionAutocomplete() {
  const cursorPos = inputEl.selectionStart ?? inputEl.value.length;
  const textBeforeCursor = inputEl.value.slice(0, cursorPos);
  const match = textBeforeCursor.match(/(?:^|\s)@([a-zA-Z]*)$/);

  if (!match) {
    hideMentionMenu();
    return;
  }

  const query = match[1].toLowerCase();
  const matches = MENTION_CANDIDATES.filter(
    (c) => c.id.startsWith(query) || c.name.toLowerCase().startsWith(query),
  );

  if (matches.length === 0) {
    hideMentionMenu();
    return;
  }

  mentionMenuEl.innerHTML = '';
  for (const candidate of matches) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'mention-item';
    item.innerHTML = `
      <span class="avatar avatar-${candidate.id}">${candidate.name[0]}</span>
      <span class="participant-info">
        <span class="participant-name">${candidate.tag}</span>
        <span class="participant-role">${candidate.role}</span>
      </span>
    `;
    item.addEventListener('click', () => {
      insertMentionIntoComposer(`${candidate.tag} `);
    });
    mentionMenuEl.appendChild(item);
  }

  mentionMenuEl.classList.remove('hidden');
}

function hideMentionMenu() {
  mentionMenuEl.classList.add('hidden');
}

document.querySelectorAll('[data-mention]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const mention = btn.getAttribute('data-mention') || '';
    insertMentionIntoComposer(mention);
  });
});

document.querySelectorAll('[data-prompt]').forEach((chip) => {
  chip.addEventListener('click', () => {
    const prompt = chip.getAttribute('data-prompt') || '';
    sendUserMessage(prompt);
  });
});

stopBtnEl.addEventListener('click', async () => {
  stopBtnEl.disabled = true;
  try {
    await fetch('/api/stop', { method: 'POST' });
  } catch {
    // Ignore network error
  } finally {
    stopBtnEl.disabled = false;
    inputEl.focus();
  }
});

resetBtnEl.addEventListener('click', async () => {
  resetBtnEl.disabled = true;
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    if (res.ok) {
      messagesById.clear();
      renderAllMessages(true);
    }
  } catch {
    // Ignore network error
  } finally {
    resetBtnEl.disabled = false;
    inputEl.focus();
  }
});

fetchInitialState().then(() => {
  connectSSE();
});
