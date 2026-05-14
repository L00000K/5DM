import { AppState } from './app.js';

export function initTextInput() {
  const btnAdd    = document.getElementById('btn-add-desc');
  const textarea  = document.getElementById('input-unit-desc');
  const descList  = document.getElementById('desc-list');

  AppState.textDescriptions = AppState.textDescriptions || [];
  AppState.siteHistory      = '';

  document.getElementById('input-site-history').addEventListener('change', e => {
    AppState.siteHistory = e.target.value.trim();
  });

  btnAdd.addEventListener('click', () => {
    const text = textarea.value.trim();
    if (!text) return;
    AppState.textDescriptions.push(text);
    textarea.value = '';
    renderDescList();
  });

  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) btnAdd.click();
  });

  function renderDescList() {
    descList.innerHTML = '';
    AppState.textDescriptions.forEach((desc, i) => {
      const item = document.createElement('div');
      item.className = 'desc-item';
      item.innerHTML = `
        <span class="desc-text">${escHtml(desc)}</span>
        <button class="file-remove" data-i="${i}" title="Remove">×</button>`;
      descList.appendChild(item);
    });
    descList.querySelectorAll('.file-remove').forEach(btn => {
      btn.addEventListener('click', e => {
        const i = parseInt(e.currentTarget.dataset.i);
        AppState.textDescriptions.splice(i, 1);
        renderDescList();
      });
    });
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
