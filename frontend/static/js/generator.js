/**
 * Combination Generator JavaScript
 */

// State
const state = {
    parameters: [],
    dependencies: [],
    currentPath: '',
    previewData: null
};

// DOM Elements cache
let elements = {};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    cacheElements();
    setupEventListeners();
    initTheme();
    addParameter(); // Add first parameter by default
    updateNamePreview();
});

/**
 * Cache DOM elements for performance
 */
function cacheElements() {
    elements = {
        // Buttons
        addParameterBtn: document.getElementById('add-parameter-btn'),
        addDependencyBtn: document.getElementById('add-dependency-btn'),
        calculateCountBtn: document.getElementById('calculate-count-btn'),
        previewBtn: document.getElementById('preview-btn'),
        generateBtn: document.getElementById('generate-btn'),
        themeToggleBtn: document.getElementById('theme-toggle-btn'),
        browsePathBtn: document.getElementById('browse-path-btn'),

        // Containers
        parameterList: document.getElementById('parameter-list'),
        dependencyList: document.getElementById('dependency-list'),
        resultsSection: document.getElementById('results-section'),
        previewTableWrapper: document.getElementById('preview-table-wrapper'),
        downloadContainer: document.getElementById('download-container'),

        // Inputs
        namePrefix: document.getElementById('name-prefix'),
        namePostfix: document.getElementById('name-postfix'),
        savePath: document.getElementById('save-path'),
        filenamePrefix: document.getElementById('filename-prefix'),

        // Results
        totalCombinations: document.getElementById('total-combinations'),
        numFiles: document.getElementById('num-files'),
        namePreview: document.getElementById('name-preview'),

        // Loading
        loading: document.getElementById('loading'),
        loadingMessage: document.getElementById('loading-message'),

        // Modal
        dirBrowserModal: document.getElementById('dir-browser-modal'),
        closeModalBtn: document.getElementById('close-modal-btn'),
        goUpBtn: document.getElementById('go-up-btn'),
        currentPathInput: document.getElementById('current-path'),
        browserLoading: document.getElementById('browser-loading'),
        dirList: document.getElementById('dir-list'),
        cancelBrowseBtn: document.getElementById('cancel-browse-btn'),
        selectDirBtn: document.getElementById('select-dir-btn')
    };
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Add parameter/dependency buttons
    elements.addParameterBtn.addEventListener('click', addParameter);
    elements.addDependencyBtn.addEventListener('click', addDependency);

    // Action buttons
    elements.calculateCountBtn.addEventListener('click', calculateCount);
    elements.previewBtn.addEventListener('click', generatePreview);
    elements.generateBtn.addEventListener('click', generateCSV);

    // Theme toggle
    elements.themeToggleBtn.addEventListener('click', toggleTheme);

    // Name configuration changes
    elements.namePrefix.addEventListener('input', updateNamePreview);
    elements.namePostfix.addEventListener('input', updateNamePreview);

    // Directory browser
    elements.browsePathBtn.addEventListener('click', openDirectoryBrowser);
    elements.closeModalBtn.addEventListener('click', closeModal);
    elements.cancelBrowseBtn.addEventListener('click', closeModal);
    elements.selectDirBtn.addEventListener('click', selectDirectory);
    elements.goUpBtn.addEventListener('click', goUpDirectory);

    // Close modal on backdrop click
    elements.dirBrowserModal.addEventListener('click', function(e) {
        if (e.target === this) closeModal();
    });
}

/**
 * Initialize theme from localStorage
 */
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        updateThemeButton(true);
    }
}

/**
 * Toggle theme between light and dark
 */
function toggleTheme() {
    const isDark = document.body.classList.contains('dark-theme');
    if (isDark) {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        localStorage.setItem('theme', 'light');
    } else {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
    }
    updateThemeButton(!isDark);
}

/**
 * Update theme button text
 */
function updateThemeButton(isDark) {
    if (isDark) {
        elements.themeToggleBtn.innerHTML = '<span class="theme-icon">&#9788;</span> Light Mode';
    } else {
        elements.themeToggleBtn.innerHTML = '<span class="theme-icon">&#9790;</span> Dark Mode';
    }
}

/**
 * Add a new parameter row
 */
function addParameter() {
    const index = state.parameters.length;
    const paramId = `param-${Date.now()}`;

    state.parameters.push({
        id: paramId,
        name: `param${index + 1}`,
        min: 0,
        max: 100,
        step: 10
    });

    const row = document.createElement('div');
    row.className = 'parameter-row';
    row.dataset.paramId = paramId;

    row.innerHTML = `
        <div class="form-group">
            <label>Name</label>
            <input type="text" class="param-name" value="param${index + 1}" placeholder="Parameter name">
        </div>
        <div class="form-group">
            <label>Min</label>
            <input type="number" class="param-min" value="0" step="any">
        </div>
        <div class="form-group">
            <label>Max</label>
            <input type="number" class="param-max" value="100" step="any">
        </div>
        <div class="form-group">
            <label>Step</label>
            <input type="number" class="param-step" value="10" step="any" min="0.0001">
        </div>
        <button class="btn btn-danger btn-remove" title="Remove parameter">&#10005;</button>
    `;

    // Setup event listeners for this row
    row.querySelector('.param-name').addEventListener('input', function() {
        updateParameterState(paramId, 'name', this.value);
        updateNamePreview();
    });
    row.querySelector('.param-min').addEventListener('input', function() {
        updateParameterState(paramId, 'min', parseFloat(this.value) || 0);
        updateNamePreview();
    });
    row.querySelector('.param-max').addEventListener('input', function() {
        updateParameterState(paramId, 'max', parseFloat(this.value) || 0);
    });
    row.querySelector('.param-step').addEventListener('input', function() {
        updateParameterState(paramId, 'step', parseFloat(this.value) || 1);
    });
    row.querySelector('.btn-remove').addEventListener('click', function() {
        removeParameter(paramId);
    });

    elements.parameterList.appendChild(row);
    updateNamePreview();
}

/**
 * Update parameter state
 */
function updateParameterState(paramId, field, value) {
    const param = state.parameters.find(p => p.id === paramId);
    if (param) {
        param[field] = value;
    }
}

/**
 * Remove a parameter
 */
function removeParameter(paramId) {
    state.parameters = state.parameters.filter(p => p.id !== paramId);
    const row = document.querySelector(`.parameter-row[data-param-id="${paramId}"]`);
    if (row) {
        row.remove();
    }
    updateNamePreview();
}

/**
 * Add a new dependency row
 */
function addDependency() {
    const depId = `dep-${Date.now()}`;

    state.dependencies.push({
        id: depId,
        type: 'condition',
        expression: ''
    });

    const row = document.createElement('div');
    row.className = 'dependency-row';
    row.dataset.depId = depId;

    row.innerHTML = `
        <div class="form-group">
            <label>Type</label>
            <select class="dep-type">
                <option value="condition">Condition</option>
                <option value="filter">Filter</option>
            </select>
        </div>
        <div class="form-group">
            <label>Expression</label>
            <input type="text" class="dep-expression" placeholder="e.g., param2 > param1">
        </div>
        <button class="btn btn-danger btn-remove" title="Remove dependency">&#10005;</button>
    `;

    // Setup event listeners
    row.querySelector('.dep-type').addEventListener('change', function() {
        updateDependencyState(depId, 'type', this.value);
    });
    row.querySelector('.dep-expression').addEventListener('input', function() {
        updateDependencyState(depId, 'expression', this.value);
    });
    row.querySelector('.btn-remove').addEventListener('click', function() {
        removeDependency(depId);
    });

    elements.dependencyList.appendChild(row);
}

/**
 * Update dependency state
 */
function updateDependencyState(depId, field, value) {
    const dep = state.dependencies.find(d => d.id === depId);
    if (dep) {
        dep[field] = value;
    }
}

/**
 * Remove a dependency
 */
function removeDependency(depId) {
    state.dependencies = state.dependencies.filter(d => d.id !== depId);
    const row = document.querySelector(`.dependency-row[data-dep-id="${depId}"]`);
    if (row) {
        row.remove();
    }
}

/**
 * Update the name preview
 */
function updateNamePreview() {
    const prefix = elements.namePrefix.value || '';
    const postfix = elements.namePostfix.value || '';

    // Group parameters by indicator prefix (text before first underscore)
    const indicatorGroups = {};
    state.parameters.forEach(param => {
        const parts = param.name.split('_');
        const indicator = parts.length > 1 ? parts[0] : param.name;
        if (!indicatorGroups[indicator]) {
            indicatorGroups[indicator] = [];
        }
        const sampleValue = param.min !== undefined ? param.min : 0;
        indicatorGroups[indicator].push(sampleValue);
    });

    // Build indicator parts: EMA[1, 10] + ADX[5, 14]
    const indicatorParts = Object.entries(indicatorGroups).map(([indicator, values]) => {
        return `${indicator}[${values.join(', ')}]`;
    });

    let paramParts = indicatorParts.join(' + ');
    if (paramParts === '') {
        paramParts = 'indicator[values]';
    }

    const preview = prefix + paramParts + postfix;
    elements.namePreview.textContent = preview;
}

/**
 * Get request data for API calls
 */
function getRequestData() {
    // Collect parameters from state
    const parameters = state.parameters.map(p => ({
        name: p.name,
        min_value: p.min,
        max_value: p.max,
        step: p.step
    }));

    // Collect dependencies (filter out empty expressions)
    const dependencies = state.dependencies
        .filter(d => d.expression.trim())
        .map(d => ({
            type: d.type,
            expression: d.expression
        }));

    return {
        parameters,
        dependencies,
        name_prefix: elements.namePrefix.value || '',
        name_postfix: elements.namePostfix.value || ''
    };
}

/**
 * Validate request data
 */
function validateRequest() {
    if (state.parameters.length === 0) {
        alert('Please add at least one parameter');
        return false;
    }

    for (const param of state.parameters) {
        if (!param.name.trim()) {
            alert('All parameters must have a name');
            return false;
        }
        if (param.step <= 0) {
            alert('Step must be greater than 0');
            return false;
        }
        if (param.min > param.max) {
            alert(`Parameter "${param.name}": min value cannot be greater than max value`);
            return false;
        }
    }

    return true;
}

/**
 * Show loading indicator
 */
function showLoading(message = 'Processing...') {
    elements.loadingMessage.textContent = message;
    elements.loading.style.display = 'flex';
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    elements.loading.style.display = 'none';
}

/**
 * Calculate combination count
 */
async function calculateCount() {
    if (!validateRequest()) return;

    showLoading('Calculating combinations...');

    try {
        const requestData = getRequestData();
        const response = await fetch('/generator/calculate-count', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to calculate count');
        }

        const data = await response.json();

        // Show results section
        elements.resultsSection.classList.add('visible');
        elements.totalCombinations.textContent = data.total_combinations.toLocaleString();
        elements.numFiles.textContent = data.num_files;

        if (data.estimated) {
            elements.totalCombinations.textContent += ' (estimated)';
        }

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Generate preview
 */
async function generatePreview() {
    if (!validateRequest()) return;

    showLoading('Generating preview...');

    try {
        const requestData = {
            ...getRequestData(),
            preview_limit: 100
        };

        const response = await fetch('/generator/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate preview');
        }

        const data = await response.json();
        state.previewData = data;

        // Show results section
        elements.resultsSection.classList.add('visible');
        elements.totalCombinations.textContent = data.total_combinations.toLocaleString();
        elements.numFiles.textContent = data.num_files;

        // Render preview table
        renderPreviewTable(data.columns, data.preview_data);

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Render preview table
 */
function renderPreviewTable(columns, data) {
    if (!data || data.length === 0) {
        elements.previewTableWrapper.innerHTML = `
            <div class="empty-state">
                <p>No combinations generated</p>
            </div>
        `;
        return;
    }

    let html = '<table class="preview-table"><thead><tr>';

    // Headers
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Rows
    data.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            html += `<td>${value !== null && value !== undefined ? value : ''}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    elements.previewTableWrapper.innerHTML = html;
}

/**
 * Generate CSV
 */
async function generateCSV() {
    if (!validateRequest()) return;

    showLoading('Generating CSV files...');

    try {
        const requestData = {
            ...getRequestData(),
            save_path: elements.savePath.value || null,
            filename_prefix: elements.filenamePrefix.value || 'combinations'
        };

        const response = await fetch('/generator/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate CSV');
        }

        const data = await response.json();

        // Show results section
        elements.resultsSection.classList.add('visible');
        elements.totalCombinations.textContent = data.total_combinations.toLocaleString();
        elements.numFiles.textContent = data.num_files;

        // Show download buttons
        renderDownloadButtons(data.filenames, data.download_urls);

        // Show success message
        let message = `Successfully generated ${data.num_files} file(s) with ${data.total_combinations.toLocaleString()} combinations.`;
        if (data.save_path) {
            message += `\n\nFiles saved to: ${data.save_path}`;
        }
        alert(message);

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Render download buttons
 */
function renderDownloadButtons(filenames, downloadUrls) {
    elements.downloadContainer.style.display = 'flex';
    elements.downloadContainer.innerHTML = '';

    filenames.forEach((filename, index) => {
        const url = downloadUrls[index];
        const btn = document.createElement('a');
        btn.className = 'download-btn';
        btn.href = url;
        btn.download = filename;
        btn.innerHTML = `<span class="download-icon">&#8595;</span> ${filename}`;
        elements.downloadContainer.appendChild(btn);
    });
}

/**
 * Open directory browser modal
 */
async function openDirectoryBrowser() {
    elements.dirBrowserModal.classList.add('show');
    state.currentPath = elements.savePath.value || '/';
    await loadDirectory(state.currentPath);
}

/**
 * Close modal
 */
function closeModal() {
    elements.dirBrowserModal.classList.remove('show');
}

/**
 * Load directory contents
 */
async function loadDirectory(path) {
    elements.browserLoading.style.display = 'flex';
    elements.dirList.innerHTML = '';
    elements.currentPathInput.value = path;

    try {
        const response = await fetch(`/optimization/browse?path=${encodeURIComponent(path)}`);

        if (!response.ok) {
            throw new Error('Failed to load directory');
        }

        const data = await response.json();
        state.currentPath = data.current_path;
        elements.currentPathInput.value = data.current_path;

        // Render directory contents (only directories)
        const directories = data.directories || [];

        if (directories.length === 0) {
            elements.dirList.innerHTML = '<li class="empty-message">No subdirectories</li>';
        } else {
            directories.forEach(item => {
                const li = document.createElement('li');
                li.className = 'file-item directory';
                li.innerHTML = `
                    <span class="file-icon">&#128193;</span>
                    <span class="file-name">${item.name}</span>
                `;
                li.addEventListener('click', () => {
                    loadDirectory(item.path);
                });
                elements.dirList.appendChild(li);
            });
        }

    } catch (error) {
        elements.dirList.innerHTML = `<li class="error-message">Error: ${error.message}</li>`;
    } finally {
        elements.browserLoading.style.display = 'none';
    }
}

/**
 * Go up one directory
 */
function goUpDirectory() {
    const parts = state.currentPath.split('/').filter(p => p);
    if (parts.length > 0) {
        parts.pop();
        const newPath = '/' + parts.join('/');
        loadDirectory(newPath || '/');
    }
}

/**
 * Select current directory
 */
function selectDirectory() {
    elements.savePath.value = state.currentPath;
    closeModal();
}
