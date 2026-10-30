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
    bubbleEl.textContent = msg.content;
    wrapEl.appendChild(bubbleEl);

    rowEl.appendChild(wrapEl);
  }

  return rowEl;
}

function updateBubbleContent(bubbleEl, msg) {
  bubbleEl.textContent = msg.content || '';
  if (msg.status === 'streaming') {
    const dots = document.createElement('span');
    dots.className = 'typing-dots';
    dots.setAttribute('aria-hidden', 'true');
    dots.innerHTML = '<span></span><span></span><span></span>';
    bubbleEl.appendChild(dots);
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

function connectSSE() {
  const es = new EventSource('/api/stream');

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
    // Browser EventSource automatically reconnects; refresh canonical state on reconnect
    setTimeout(() => {
      fetchInitialState().catch(() => {});
    }, 1000);
  };
}

async function sendUserMessage(rawText) {
  const content = rawText.trim();
  if (!content) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  hideMentionMenu();
  inputEl.focus();

  await fetch('/api/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
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
  if (atMatch) {
    const replaceStart = before.lastIndexOf('@');
    inputEl.value = before.slice(0, replaceStart) + mentionText + after;
  } else {
    const needsSpace = before.length > 0 && !/\s$/.test(before);
    inputEl.value = before + (needsSpace ? ' ' : '') + mentionText + after;
  }

  hideMentionMenu();
  inputEl.focus();
  const cursorPos = inputEl.value.length;
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
  await fetch('/api/stop', { method: 'POST' });
  inputEl.focus();
});

resetBtnEl.addEventListener('click', async () => {
  await fetch('/api/reset', { method: 'POST' });
  messagesById.clear();
  renderAllMessages(true);
  inputEl.focus();
});

fetchInitialState().then(() => {
  connectSSE();
});
