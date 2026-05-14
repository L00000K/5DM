const STORAGE_KEY = 'geomodel-api-key';

export function getApiKey() {
  return sessionStorage.getItem(STORAGE_KEY);
}

export function setApiKey(key) {
  if (key) sessionStorage.setItem(STORAGE_KEY, key);
  else sessionStorage.removeItem(STORAGE_KEY);
}

export function clearApiKey() {
  sessionStorage.removeItem(STORAGE_KEY);
}

export function initApiKeyModal() {
  const modal    = document.getElementById('modal-api-key');
  const btnOpen  = document.getElementById('btn-api-key');
  const btnSave  = document.getElementById('btn-save-key');
  const btnDemo  = document.getElementById('btn-demo-mode');
  const inputKey = document.getElementById('input-api-key');
  const status   = document.getElementById('api-key-status');
  const backdrop = modal.querySelector('.modal-backdrop');

  function open() {
    const existing = getApiKey();
    if (existing) {
      inputKey.value = existing;
      status.textContent = '✓ Key set for this session';
      status.style.color = 'var(--green)';
    } else {
      inputKey.value = '';
      status.textContent = '';
    }
    modal.hidden = false;
    inputKey.focus();
  }

  function close() {
    modal.hidden = true;
  }

  btnOpen.addEventListener('click', open);
  backdrop.addEventListener('click', close);

  btnSave.addEventListener('click', () => {
    const val = inputKey.value.trim();
    if (!val.startsWith('sk-ant')) {
      status.textContent = '⚠ Key should start with sk-ant-…';
      status.style.color = 'var(--accent)';
      return;
    }
    setApiKey(val);
    status.textContent = '✓ Saved for this session';
    status.style.color = 'var(--green)';
    window.dispatchEvent(new CustomEvent('geomodel:api-key-set', { detail: { key: val } }));
    setTimeout(close, 800);
  });

  btnDemo.addEventListener('click', () => {
    clearApiKey();
    status.textContent = 'Demo mode — no key required';
    status.style.color = 'var(--text-dim)';
    window.dispatchEvent(new CustomEvent('geomodel:api-key-set', { detail: { key: null } }));
    setTimeout(close, 600);
  });

  inputKey.addEventListener('keydown', e => {
    if (e.key === 'Enter') btnSave.click();
    if (e.key === 'Escape') close();
  });

  if (!getApiKey()) {
    // Auto-show on first load after a short delay
    setTimeout(open, 500);
  }
}
