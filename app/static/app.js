const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const modalitySelect = document.getElementById('modality-select');
const uploadZone = document.getElementById('upload-zone');
const browseButton = document.getElementById('browse-button');
const clearButton = document.getElementById('clear-button');
const predictButton = document.getElementById('predict-button');
const resultEmpty = document.getElementById('result-empty');
const resultPanel = document.getElementById('result-panel');
const previewImage = document.getElementById('preview-image');
const resultModality = document.getElementById('result-modality');
const resultClass = document.getElementById('result-class');
const resultConfidence = document.getElementById('result-confidence');
const resultError = document.getElementById('result-error');
const resultNote = document.getElementById('result-note');
const anotherImageButton = document.getElementById('another-image-button');
const modelList = document.getElementById('model-list');

let selectedFile = null;
let previewUrl = null;

function setUploadState(file) {
  if (!file) {
    resultEmpty.classList.remove('hidden');
    resultPanel.classList.add('hidden');
    return;
  }

  resultEmpty.classList.add('hidden');
  resultPanel.classList.remove('hidden');

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
}

function clearResultState() {
  resultError.classList.add('hidden');
  resultError.textContent = '';
  resultNote.textContent = '';
  resultModality.textContent = '—';
  resultClass.textContent = '—';
  resultConfidence.textContent = '—';
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewImage.removeAttribute('src');
  imageInput.value = '';
  selectedFile = null;
  resultEmpty.classList.remove('hidden');
  resultPanel.classList.add('hidden');
}

function renderModelCards(modalities) {
  modelList.innerHTML = '';

  modalities.forEach((modality) => {
    const card = document.createElement('article');
    card.className = 'model-card';

    const statusClass = modality.status === 'available' ? 'available' : 'unavailable';
    const metricHtml = modality.metrics
      ? `
        <ul>
          <li>Accuracy: ${modality.metrics.test_accuracy.toFixed(4)}</li>
          <li>Macro F1: ${modality.metrics.test_macro_f1.toFixed(4)}</li>
          <li>ROC-AUC: ${modality.metrics.roc_auc.toFixed(4)}</li>
        </ul>
      `
      : '<p>Excluded from reliable prediction workflows.</p>';

    card.innerHTML = `
      <span class="status ${statusClass}">${modality.status}</span>
      <h4>${modality.name}</h4>
      <p>${modality.description}</p>
      ${metricHtml}
      <p>${modality.note}</p>
    `;
    modelList.appendChild(card);
  });
}

async function loadModelInfo() {
  const response = await fetch('/api/model-info');
  const data = await response.json();
  renderModelCards(data.modalities);
}

browseButton.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (file) {
    selectedFile = file;
    setUploadState(file);
  }
});

['dragenter', 'dragover'].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove('dragover');
  });
});

uploadZone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    imageInput.files = event.dataTransfer.files;
    setUploadState(file);
  }
});

clearButton.addEventListener('click', () => {
  clearResultState();
});

anotherImageButton.addEventListener('click', () => {
  clearResultState();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!selectedFile) {
    resultError.classList.remove('hidden');
    resultError.textContent = 'Please select an image before running inference.';
    return;
  }

  resultEmpty.classList.add('hidden');
  resultPanel.classList.remove('hidden');
  resultError.classList.add('hidden');
  resultError.textContent = '';
  resultNote.textContent = 'Running prediction...';
  predictButton.disabled = true;
  predictButton.textContent = 'Predicting...';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(`/api/predict?modality=${encodeURIComponent(modalitySelect.value)}`, {
      method: 'POST',
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok || !payload.success) {
      resultError.classList.remove('hidden');
      resultError.textContent = payload.error || 'Prediction failed.';
      resultNote.textContent = 'Please check the file and try again.';
      return;
    }

    resultModality.textContent = payload.modality;
    resultClass.textContent = payload.predicted_class;
    resultConfidence.textContent = `${payload.model_confidence_percentage}%`;
    resultNote.textContent = payload.disclaimer;
  } catch (error) {
    resultError.classList.remove('hidden');
    resultError.textContent = `Prediction request failed: ${error.message}`;
    resultNote.textContent = 'The server may be unavailable or the image could not be processed.';
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = 'Run prediction';
  }
});

window.addEventListener('DOMContentLoaded', async () => {
  await loadModelInfo();
  clearResultState();
});
