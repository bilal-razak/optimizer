/**
 * Optimizer Dashboard - Step-by-Step Workflow
 * Frontend JavaScript for API interaction and visualization
 */

const API_BASE = '';

// State management
const state = {
    sessionId: null,
    currentStep: 1,
    columns: [],
    strategyParams: [],
    availableMetrics: [],
    shortlistApplied: false,
    heatmaps: [],           // All generated heatmaps
    filteredHeatmaps: [],   // Filtered heatmaps based on navigation
    heatmapIndex: 0,
    heatmapConstParam: null,    // The const param used for generation
    heatmapConstValues: [],     // Unique const values in generated heatmaps
    selectedConstValues: [],    // Currently selected const values for filtering
    heatmapMetrics: [],         // Unique metrics in generated heatmaps
    availableConfigs: [],
    kmeansAllStats: {},         // K-Means stats for all metrics
    hdbscanFinalResult: null,   // HDBSCAN final results for view toggling
    isDarkTheme: false,
    browser: {
        currentPath: '',
        selectedFile: null
    }
};

// DOM Elements
const elements = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    cacheElements();
    setupEventListeners();
    initTheme();
    showStep(1);
});

function cacheElements() {
    // Step panels
    elements.stepPanels = document.querySelectorAll('.step-panel');
    elements.stepItems = document.querySelectorAll('.step-item');

    // Theme toggle
    elements.themeToggleBtn = document.getElementById('theme-toggle-btn');

    // Step 1
    elements.csvPath = document.getElementById('csv-path');
    elements.browseBtn = document.getElementById('browse-btn');
    elements.loadColumnsBtn = document.getElementById('load-columns-btn');
    elements.columnSelectionSection = document.getElementById('column-selection-section');
    elements.columnSearch = document.getElementById('column-search');
    elements.selectedParamsTags = document.getElementById('selected-params-tags');
    elements.paramCount = document.getElementById('param-count');
    elements.paramCheckboxes = document.getElementById('param-checkboxes');
    elements.runStep1Btn = document.getElementById('run-step1-btn');
    elements.step1Results = document.getElementById('step1-results');
    elements.approveStep1Btn = document.getElementById('approve-step1-btn');

    // Table filtering
    elements.tableFilter = document.getElementById('table-filter');
    elements.tableColumnFilter = document.getElementById('table-column-filter');

    // Step 2
    elements.enableShortlist = document.getElementById('enable-shortlist');
    elements.shortlistConditions = document.getElementById('shortlist-conditions');
    elements.addConditionBtn = document.getElementById('add-condition-btn');
    elements.applyShortlistBtn = document.getElementById('apply-shortlist-btn');
    elements.shortlistStatus = document.getElementById('shortlist-status');
    elements.heatmapXParam = document.getElementById('heatmap-x-param');
    elements.heatmapYParam = document.getElementById('heatmap-y-param');
    elements.heatmapConstParam = document.getElementById('heatmap-const-param');
    elements.showShortlisted = document.getElementById('show-shortlisted');
    elements.generateHeatmapBtn = document.getElementById('generate-heatmap-btn');
    elements.step2Results = document.getElementById('step2-results');
    elements.approveStep2Btn = document.getElementById('approve-step2-btn');
    // Navigation elements
    elements.navMetric = document.getElementById('nav-metric');
    elements.navConstValueGroup = document.getElementById('nav-const-value-group');
    elements.constValueToggle = document.getElementById('const-value-toggle');
    elements.constValueDisplay = document.getElementById('const-value-display');
    elements.constValueDropdown = document.getElementById('const-value-dropdown');
    elements.constValueOptions = document.getElementById('const-value-options');
    elements.constSelectAll = document.getElementById('const-select-all');
    elements.constClearAll = document.getElementById('const-clear-all');

    // Step 3-7 elements
    elements.runStep3Btn = document.getElementById('run-step3-btn');
    elements.step3Results = document.getElementById('step3-results');
    elements.approveStep3Btn = document.getElementById('approve-step3-btn');
    elements.kmeansK = document.getElementById('kmeans-k');
    elements.runStep4Btn = document.getElementById('run-step4-btn');
    elements.step4Results = document.getElementById('step4-results');
    elements.approveStep4Btn = document.getElementById('approve-step4-btn');
    elements.kmeansStatsMetric = document.getElementById('kmeans-stats-metric');
    elements.hdbscanMinSizes = document.getElementById('hdbscan-min-sizes');
    elements.hdbscanMinSamples = document.getElementById('hdbscan-min-samples');
    elements.hdbscanThreshold = document.getElementById('hdbscan-threshold');
    elements.rankingMetric = document.getElementById('ranking-metric');
    elements.runStep5Btn = document.getElementById('run-step5-btn');
    elements.step5Results = document.getElementById('step5-results');
    elements.approveStep5Btn = document.getElementById('approve-step5-btn');
    elements.finalMinClusterSize = document.getElementById('final-min-cluster-size');
    elements.finalMinSamples = document.getElementById('final-min-samples');
    elements.runStep6Btn = document.getElementById('run-step6-btn');
    elements.step6Results = document.getElementById('step6-results');
    elements.approveStep6Btn = document.getElementById('approve-step6-btn');
    elements.numBestClusters = document.getElementById('num-best-clusters');
    elements.runStep7Btn = document.getElementById('run-step7-btn');
    elements.step7Results = document.getElementById('step7-results');
    elements.restartBtn = document.getElementById('restart-btn');

    // Loading & Modal
    elements.loading = document.getElementById('loading');
    elements.loadingMessage = document.getElementById('loading-message');
    elements.modal = document.getElementById('file-browser-modal');
    elements.closeModalBtn = document.getElementById('close-modal-btn');
    elements.goUpBtn = document.getElementById('go-up-btn');
    elements.currentPathInput = document.getElementById('current-path');
    elements.goToPathBtn = document.getElementById('go-to-path-btn');
    elements.browserLoading = document.getElementById('browser-loading');
    elements.fileList = document.getElementById('file-list');
    elements.selectedFileContainer = document.getElementById('selected-file-container');
    elements.selectedFileName = document.getElementById('selected-file-name');
    elements.cancelBrowseBtn = document.getElementById('cancel-browse-btn');
    elements.confirmBrowseBtn = document.getElementById('confirm-browse-btn');
}

function setupEventListeners() {
    // Theme toggle
    if (elements.themeToggleBtn) {
        elements.themeToggleBtn.addEventListener('click', toggleTheme);
    }

    // Back buttons
    document.querySelectorAll('.btn-back').forEach(btn => {
        btn.addEventListener('click', function() {
            const backStep = parseInt(this.dataset.back);
            showStep(backStep);
        });
    });

    // File browser
    elements.browseBtn.addEventListener('click', openFileBrowser);
    elements.closeModalBtn.addEventListener('click', closeFileBrowser);
    elements.cancelBrowseBtn.addEventListener('click', closeFileBrowser);
    elements.confirmBrowseBtn.addEventListener('click', confirmFileSelection);
    elements.goUpBtn.addEventListener('click', goToParentDirectory);
    elements.goToPathBtn.addEventListener('click', () => browseDirectory(elements.currentPathInput.value));
    elements.modal.addEventListener('click', (e) => {
        if (e.target === elements.modal) closeFileBrowser();
    });

    // Step 1
    elements.loadColumnsBtn.addEventListener('click', loadColumns);
    if (elements.columnSearch) {
        elements.columnSearch.addEventListener('input', filterColumns);
    }
    elements.runStep1Btn.addEventListener('click', runStep1);
    elements.approveStep1Btn.addEventListener('click', () => showStep(2));

    // Table filtering
    if (elements.tableFilter) {
        elements.tableFilter.addEventListener('input', filterTable);
    }
    if (elements.tableColumnFilter) {
        elements.tableColumnFilter.addEventListener('change', filterTable);
    }

    // Step 2
    elements.enableShortlist.addEventListener('change', (e) => {
        elements.shortlistConditions.style.display = e.target.checked ? 'block' : 'none';
    });
    elements.addConditionBtn.addEventListener('click', addCondition);
    elements.shortlistConditions.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove-condition')) {
            e.target.closest('.condition-row').remove();
        }
    });
    elements.applyShortlistBtn.addEventListener('click', applyShortlist);
    elements.generateHeatmapBtn.addEventListener('click', generateHeatmaps);
    elements.approveStep2Btn.addEventListener('click', () => showStep(3));

    // Heatmap navigation
    document.getElementById('prev-heatmap').addEventListener('click', () => navigateHeatmap(-1));
    document.getElementById('next-heatmap').addEventListener('click', () => navigateHeatmap(1));
    elements.navMetric.addEventListener('change', filterHeatmaps);

    // Multi-select dropdown for const values
    elements.constValueToggle.addEventListener('click', toggleConstValueDropdown);
    elements.constSelectAll.addEventListener('click', selectAllConstValues);
    elements.constClearAll.addEventListener('click', clearAllConstValues);

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.multi-select-container') && !e.target.closest('.multi-select-dropdown')) {
            elements.constValueDropdown.style.display = 'none';
        }
    });

    // Steps 3-7
    elements.runStep3Btn.addEventListener('click', runStep3);
    elements.approveStep3Btn.addEventListener('click', () => showStep(4));
    elements.runStep4Btn.addEventListener('click', runStep4);
    elements.approveStep4Btn.addEventListener('click', () => showStep(5));
    elements.runStep5Btn.addEventListener('click', runStep5);
    elements.approveStep5Btn.addEventListener('click', () => showStep(6));
    elements.runStep6Btn.addEventListener('click', runStep6);
    elements.approveStep6Btn.addEventListener('click', () => showStep(7));
    elements.runStep7Btn.addEventListener('click', runStep7);
    elements.restartBtn.addEventListener('click', restart);

    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            window.dispatchEvent(new Event('resize'));
        });
    });
}

// ==================== Theme Toggle ====================

function initTheme() {
    // Check for saved preference or default to light
    const savedTheme = localStorage.getItem('dashboardTheme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        document.body.classList.remove('light-theme');
        state.isDarkTheme = true;
        updateThemeButton();
    } else {
        document.body.classList.add('light-theme');
        document.body.classList.remove('dark-theme');
        state.isDarkTheme = false;
        updateThemeButton();
    }
}

function toggleTheme() {
    state.isDarkTheme = !state.isDarkTheme;
    if (state.isDarkTheme) {
        document.body.classList.add('dark-theme');
        document.body.classList.remove('light-theme');
        localStorage.setItem('dashboardTheme', 'dark');
    } else {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        localStorage.setItem('dashboardTheme', 'light');
    }
    updateThemeButton();

    // Re-render any visible charts with new theme
    reRenderCharts();
}

function updateThemeButton() {
    if (elements.themeToggleBtn) {
        if (state.isDarkTheme) {
            elements.themeToggleBtn.innerHTML = '<span class="theme-icon">&#9788;</span> Light Mode';
        } else {
            elements.themeToggleBtn.innerHTML = '<span class="theme-icon">&#9790;</span> Dark Mode';
        }
    }
}

function reRenderCharts() {
    // Re-render any visible heatmaps
    if (state.heatmaps.length > 0) {
        renderHeatmap(state.heatmaps[state.heatmapIndex]);
    }
}

function getChartColors() {
    if (state.isDarkTheme) {
        return {
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            gridcolor: '#334155',
            fontcolor: '#f1f5f9'
        };
    } else {
        return {
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            gridcolor: '#e2e8f0',
            fontcolor: '#1e293b'
        };
    }
}

// ==================== Step Navigation ====================

function showStep(stepNum) {
    state.currentStep = stepNum;
    elements.stepPanels.forEach(p => p.classList.remove('active'));
    document.getElementById(`step-${stepNum}-panel`).classList.add('active');
    elements.stepItems.forEach(item => {
        const step = parseInt(item.dataset.step);
        item.classList.remove('active', 'completed');
        if (step < stepNum) item.classList.add('completed');
        else if (step === stepNum) item.classList.add('active');
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function restart() {
    if (state.sessionId) {
        fetch(`${API_BASE}/steps/session/${state.sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    state.sessionId = null;
    state.shortlistApplied = false;
    state.heatmaps = [];
    state.heatmapIndex = 0;
    state.strategyParams = [];
    document.querySelectorAll('.step-results').forEach(el => el.style.display = 'none');
    showStep(1);
}

// ==================== File Browser ====================

function openFileBrowser() {
    elements.modal.classList.add('show');
    state.browser.selectedFile = null;
    elements.selectedFileContainer.style.display = 'none';
    elements.confirmBrowseBtn.disabled = true;
    browseDirectory('');
}

function closeFileBrowser() {
    elements.modal.classList.remove('show');
}

async function browseDirectory(path) {
    try {
        elements.browserLoading.style.display = 'flex';
        elements.fileList.innerHTML = '';
        const url = path ? `${API_BASE}/optimization/browse?path=${encodeURIComponent(path)}` : `${API_BASE}/optimization/browse`;
        const response = await fetch(url);
        if (!response.ok) throw new Error((await response.json()).detail || 'Failed');
        const data = await response.json();
        state.browser.currentPath = data.current_path;
        state.browser.parentPath = data.parent_path;
        elements.currentPathInput.value = data.current_path;
        renderFileList(data.directories, data.files);
    } catch (error) {
        elements.fileList.innerHTML = `<li class="error-message">Error: ${error.message}</li>`;
    } finally {
        elements.browserLoading.style.display = 'none';
    }
}

function renderFileList(directories, files) {
    elements.fileList.innerHTML = '';
    directories.forEach(dir => {
        const li = document.createElement('li');
        li.className = 'file-item directory';
        li.innerHTML = `<span class="file-icon">&#128193;</span><span class="file-name">${dir.name}</span>`;
        li.addEventListener('click', () => browseDirectory(dir.path));
        elements.fileList.appendChild(li);
    });
    files.forEach(file => {
        const li = document.createElement('li');
        li.className = 'file-item file';
        li.innerHTML = `<span class="file-icon">&#128196;</span><span class="file-name">${file.name}</span><span class="file-size">${file.size}</span>`;
        li.addEventListener('click', function() { selectFile(file, this); });
        elements.fileList.appendChild(li);
    });
    if (directories.length === 0 && files.length === 0) {
        elements.fileList.innerHTML = '<li class="empty-message">No CSV files found</li>';
    }
}

function selectFile(file, element) {
    document.querySelectorAll('.file-item.selected').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');
    state.browser.selectedFile = file;
    elements.selectedFileContainer.style.display = 'flex';
    elements.selectedFileName.textContent = file.name;
    elements.confirmBrowseBtn.disabled = false;
}

function goToParentDirectory() {
    if (state.browser.parentPath) browseDirectory(state.browser.parentPath);
}

function confirmFileSelection() {
    if (state.browser.selectedFile) {
        elements.csvPath.value = state.browser.selectedFile.path;
        elements.loadColumnsBtn.disabled = false;
        closeFileBrowser();
    }
}

// ==================== Utilities ====================

function showLoading(message) {
    elements.loadingMessage.textContent = message || 'Processing...';
    elements.loading.style.display = 'flex';
}

function hideLoading() {
    elements.loading.style.display = 'none';
}

async function apiCall(endpoint, method = 'GET', data = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (data) options.body = JSON.stringify(data);
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API request failed');
    }
    return response.json();
}

// ==================== Step 1: Load Data ====================

async function loadColumns() {
    const csvPath = elements.csvPath.value.trim();
    if (!csvPath) return alert('Please select a CSV file first');

    try {
        elements.loadColumnsBtn.disabled = true;
        elements.loadColumnsBtn.textContent = 'Loading...';
        const data = await apiCall(`/optimization/columns?csv_path=${encodeURIComponent(csvPath)}`);
        state.columns = data.columns;
        populateColumnSelectors();
        elements.columnSelectionSection.style.display = 'block';
        elements.runStep1Btn.disabled = true;
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        elements.loadColumnsBtn.disabled = false;
        elements.loadColumnsBtn.textContent = 'Load Columns';
    }
}

function populateColumnSelectors() {
    // Strategy param checkboxes with improved UI
    elements.paramCheckboxes.innerHTML = '';
    state.strategyParams = [];
    updateSelectedParamsTags();

    state.columns.forEach(col => {
        const item = document.createElement('div');
        item.className = 'column-item';
        item.dataset.column = col;
        item.innerHTML = `<input type="checkbox" value="${col}" class="param-checkbox"> <span>${col}</span>`;

        item.addEventListener('click', (e) => {
            if (e.target.type !== 'checkbox') {
                const checkbox = item.querySelector('.param-checkbox');
                checkbox.checked = !checkbox.checked;
                checkbox.dispatchEvent(new Event('change'));
            }
        });

        const checkbox = item.querySelector('.param-checkbox');
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                if (state.strategyParams.length < 4) {
                    state.strategyParams.push(col);
                    item.classList.add('selected');
                } else {
                    checkbox.checked = false;
                    alert('Maximum 4 strategy parameters allowed');
                    return;
                }
            } else {
                state.strategyParams = state.strategyParams.filter(p => p !== col);
                item.classList.remove('selected');
            }
            updateSelectedParamsTags();
            updateRunButtonState();
        });

        elements.paramCheckboxes.appendChild(item);
    });
}

function updateSelectedParamsTags() {
    if (!elements.selectedParamsTags) return;

    elements.selectedParamsTags.innerHTML = '';
    state.strategyParams.forEach(param => {
        const tag = document.createElement('span');
        tag.className = 'param-tag';
        tag.innerHTML = `${param} <span class="remove-tag" data-param="${param}">&times;</span>`;
        tag.querySelector('.remove-tag').addEventListener('click', (e) => {
            e.stopPropagation();
            removeParam(param);
        });
        elements.selectedParamsTags.appendChild(tag);
    });

    if (elements.paramCount) {
        elements.paramCount.textContent = `${state.strategyParams.length}/4`;
    }
}

function removeParam(param) {
    state.strategyParams = state.strategyParams.filter(p => p !== param);
    const item = document.querySelector(`.column-item[data-column="${param}"]`);
    if (item) {
        item.classList.remove('selected');
        item.querySelector('.param-checkbox').checked = false;
    }
    updateSelectedParamsTags();
    updateRunButtonState();
}

function updateRunButtonState() {
    const count = state.strategyParams.length;
    elements.runStep1Btn.disabled = count < 2 || count > 4;
}

function filterColumns() {
    const searchTerm = elements.columnSearch.value.toLowerCase();
    document.querySelectorAll('.column-item').forEach(item => {
        const colName = item.dataset.column.toLowerCase();
        if (colName.includes(searchTerm)) {
            item.classList.remove('hidden');
        } else {
            item.classList.add('hidden');
        }
    });
}

async function runStep1() {
    try {
        showLoading('Loading data...');
        const request = {
            csv_path: elements.csvPath.value.trim(),
            strategy_params: state.strategyParams
        };

        const result = await apiCall('/steps/load-data', 'POST', request);
        state.sessionId = result.session_id;
        state.availableMetrics = result.available_metrics;

        // Display results
        document.getElementById('step1-num-rows').textContent = result.num_rows;
        document.getElementById('step1-num-cols').textContent = result.num_columns;
        document.getElementById('step1-strategy-params').textContent = result.strategy_params.length;

        // Render head table
        renderDataTable('data-head-table', result.head_data, result.strategy_params);

        // Populate column filter dropdown
        populateColumnFilterDropdown(result.head_data);

        // Render column info table
        renderColumnInfoTable('column-info-table', result.column_info);

        // Populate heatmap dropdowns
        populateHeatmapSelectors(result.strategy_params);

        elements.step1Results.style.display = 'block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function renderDataTable(containerId, data, highlightCols = [], enableSorting = true) {
    const container = document.getElementById(containerId);
    if (!data || data.length === 0) {
        container.innerHTML = '<p>No data</p>';
        return;
    }
    const columns = Object.keys(data[0]);

    // Store original data for sorting
    container._tableData = [...data];
    container._sortColumn = null;
    container._sortDirection = 'asc';
    container._highlightCols = highlightCols;

    let html = '<table><thead><tr>';
    columns.forEach(col => {
        const cls = highlightCols.includes(col) ? 'highlight-col' : '';
        const sortClass = enableSorting ? 'sortable-header' : '';
        html += `<th class="${cls} ${sortClass}" data-column="${col}">${col} ${enableSorting ? '<span class="sort-indicator"></span>' : ''}</th>`;
    });
    html += '</tr></thead><tbody>';
    data.forEach((row, rowIndex) => {
        html += `<tr data-row-index="${rowIndex}">`;
        columns.forEach(col => {
            const val = row[col];
            const display = val === null ? '' : (typeof val === 'number' ? val.toFixed(4) : val);
            html += `<td data-column="${col}" data-value="${val}">${display}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;

    // Add sorting event listeners
    if (enableSorting) {
        container.querySelectorAll('.sortable-header').forEach(header => {
            header.addEventListener('click', () => {
                const column = header.dataset.column;
                sortTable(containerId, column);
            });
        });
    }
}

function sortTable(containerId, column) {
    const container = document.getElementById(containerId);
    const data = container._tableData;
    const highlightCols = container._highlightCols || [];

    // Toggle sort direction
    if (container._sortColumn === column) {
        container._sortDirection = container._sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        container._sortColumn = column;
        container._sortDirection = 'asc';
    }

    const direction = container._sortDirection;

    // Sort data
    const sortedData = [...data].sort((a, b) => {
        let valA = a[column];
        let valB = b[column];

        // Handle null values
        if (valA === null || valA === undefined) return direction === 'asc' ? 1 : -1;
        if (valB === null || valB === undefined) return direction === 'asc' ? -1 : 1;

        // Numeric comparison
        if (typeof valA === 'number' && typeof valB === 'number') {
            return direction === 'asc' ? valA - valB : valB - valA;
        }

        // String comparison
        valA = String(valA).toLowerCase();
        valB = String(valB).toLowerCase();
        if (valA < valB) return direction === 'asc' ? -1 : 1;
        if (valA > valB) return direction === 'asc' ? 1 : -1;
        return 0;
    });

    // Re-render table body
    const columns = Object.keys(data[0]);
    let html = '';
    sortedData.forEach((row, rowIndex) => {
        html += `<tr data-row-index="${rowIndex}">`;
        columns.forEach(col => {
            const val = row[col];
            const display = val === null ? '' : (typeof val === 'number' ? val.toFixed(4) : val);
            html += `<td data-column="${col}" data-value="${val}">${display}</td>`;
        });
        html += '</tr>';
    });
    container.querySelector('tbody').innerHTML = html;

    // Update sort indicators
    container.querySelectorAll('.sortable-header').forEach(header => {
        const indicator = header.querySelector('.sort-indicator');
        header.classList.remove('sorted-asc', 'sorted-desc');
        if (header.dataset.column === column) {
            header.classList.add(direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
            indicator.textContent = direction === 'asc' ? ' ▲' : ' ▼';
        } else {
            indicator.textContent = '';
        }
    });
}

function populateColumnFilterDropdown(data) {
    if (!elements.tableColumnFilter || !data || data.length === 0) return;

    const columns = Object.keys(data[0]);
    elements.tableColumnFilter.innerHTML = '<option value="">All Columns</option>';
    columns.forEach(col => {
        elements.tableColumnFilter.innerHTML += `<option value="${col}">${col}</option>`;
    });
}

function filterTable() {
    const searchTerm = elements.tableFilter ? elements.tableFilter.value.toLowerCase() : '';
    const columnFilter = elements.tableColumnFilter ? elements.tableColumnFilter.value : '';

    const table = document.querySelector('#data-head-table table');
    if (!table) return;

    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(row => {
        let visible = false;
        const cells = row.querySelectorAll('td');

        cells.forEach(cell => {
            const cellColumn = cell.dataset.column;
            const cellValue = cell.textContent.toLowerCase();

            // Check column filter
            if (columnFilter && cellColumn !== columnFilter) return;

            // Check search term
            if (searchTerm === '' || cellValue.includes(searchTerm)) {
                visible = true;
            }
        });

        row.style.display = visible ? '' : 'none';
    });
}

function renderColumnInfoTable(containerId, columnInfo) {
    const container = document.getElementById(containerId);
    let html = '<table><thead><tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Null</th></tr></thead><tbody>';
    columnInfo.forEach(info => {
        html += `<tr><td>${info.name}</td><td>${info.dtype}</td><td>${info.non_null_count}</td><td>${info.null_count}</td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

function populateHeatmapSelectors(strategyParams) {
    // X and Y param dropdowns
    [elements.heatmapXParam, elements.heatmapYParam].forEach(select => {
        select.innerHTML = '';
        strategyParams.forEach(p => {
            select.innerHTML += `<option value="${p}">${p}</option>`;
        });
    });
    // Set different defaults
    if (strategyParams.length >= 2) {
        elements.heatmapXParam.value = strategyParams[0];
        elements.heatmapYParam.value = strategyParams[1];
    }

    // Constant param dropdown
    elements.heatmapConstParam.innerHTML = '<option value="">None - Single heatmap per metric</option>';
    strategyParams.forEach(p => {
        elements.heatmapConstParam.innerHTML += `<option value="${p}">${p}</option>`;
    });
}

// ==================== Step 2: Heatmaps ====================

function addCondition() {
    const row = document.createElement('div');
    row.className = 'condition-row';
    row.innerHTML = `
        <select class="condition-metric">
            <option value="sharpe_ratio">Sharpe Ratio</option>
            <option value="sortino_ratio">Sortino Ratio</option>
            <option value="profit_factor">Profit Factor</option>
            <option value="total_pnl">Total PnL</option>
            <option value="win_ratio">Win Ratio</option>
            <option value="max_draw_down">Max Drawdown</option>
        </select>
        <select class="condition-operator">
            <option value=">">&gt;</option>
            <option value=">=">&ge;</option>
            <option value="<">&lt;</option>
            <option value="<=">&le;</option>
        </select>
        <input type="number" class="condition-value" step="0.1" value="1.0">
        <button class="btn btn-small btn-danger remove-condition">&#10005;</button>
    `;
    elements.shortlistConditions.insertBefore(row, document.querySelector('.condition-actions'));
}

async function applyShortlist() {
    try {
        showLoading('Applying shortlist...');
        const conditions = [];
        document.querySelectorAll('#shortlist-conditions .condition-row').forEach(row => {
            conditions.push({
                metric: row.querySelector('.condition-metric').value,
                operator: row.querySelector('.condition-operator').value,
                value: parseFloat(row.querySelector('.condition-value').value)
            });
        });

        const request = {
            session_id: state.sessionId,
            shortlist_config: { enabled: true, conditions }
        };

        const result = await apiCall('/steps/shortlist', 'POST', request);
        state.shortlistApplied = result.shortlist_applied;

        if (result.shortlist_applied) {
            elements.shortlistStatus.textContent = `Shortlist applied: ${result.num_variants_after_shortlist} variants match`;
            elements.shortlistStatus.style.display = 'block';
            elements.showShortlisted.disabled = false;
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function generateHeatmaps() {
    try {
        showLoading('Generating all heatmaps...');

        const constParam = elements.heatmapConstParam.value || null;
        state.heatmapConstParam = constParam;

        const request = {
            session_id: state.sessionId,
            x_param: elements.heatmapXParam.value,
            y_param: elements.heatmapYParam.value,
            const_param: constParam,
            const_values: null,  // Generate for ALL values
            metrics: null,       // Generate for ALL metrics
            show_shortlisted: elements.showShortlisted.checked && state.shortlistApplied
        };

        const result = await apiCall('/steps/heatmap', 'POST', request);

        state.heatmaps = result.heatmaps;
        state.filteredHeatmaps = result.heatmaps;
        state.heatmapIndex = 0;

        // Extract unique metrics and const values from heatmaps for navigation
        extractHeatmapMetadata(result.heatmaps);

        // Populate navigation dropdowns
        populateNavigationDropdowns();

        document.getElementById('step2-num-heatmaps').textContent = result.num_heatmaps;

        if (result.heatmaps.length > 0) {
            renderHeatmap(state.filteredHeatmaps[0]);
            updateHeatmapCounter();
        } else {
            document.getElementById('heatmap-container').innerHTML = '<p>No heatmaps generated. Check your parameters.</p>';
        }

        elements.step2Results.style.display = 'block';
        elements.approveStep2Btn.style.display = 'block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function extractHeatmapMetadata(heatmaps) {
    // Extract unique metrics and const values from heatmap metadata
    const metrics = new Set();
    const constValues = new Set();

    heatmaps.forEach(hm => {
        // Use metadata if available (from backend)
        if (hm._metadata) {
            if (hm._metadata.metric) metrics.add(hm._metadata.metric);
            if (hm._metadata.const_value) constValues.add(hm._metadata.const_value);
        } else {
            // Fallback: parse from title
            if (hm.layout && hm.layout.title) {
                const titleText = typeof hm.layout.title === 'string' ? hm.layout.title : hm.layout.title.text || '';
                if (titleText.includes('<br>')) {
                    const parts = titleText.split('<br>');
                    if (parts[0]) constValues.add(parts[0].trim());
                    if (parts[1]) metrics.add(parts[1].trim());
                } else {
                    metrics.add(titleText.trim());
                }
            }
        }
    });

    state.heatmapMetrics = Array.from(metrics).sort();
    state.heatmapConstValues = Array.from(constValues).sort((a, b) => {
        // Try to sort numerically if possible
        const numA = parseFloat(a);
        const numB = parseFloat(b);
        if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
        return a.localeCompare(b);
    });
}

function populateNavigationDropdowns() {
    // Populate metric dropdown
    elements.navMetric.innerHTML = '<option value="">All Metrics</option>';
    state.heatmapMetrics.forEach(metric => {
        elements.navMetric.innerHTML += `<option value="${metric}">${metric}</option>`;
    });

    // Populate const value multi-select (only if const param was used)
    if (state.heatmapConstParam && state.heatmapConstValues.length > 0) {
        elements.navConstValueGroup.style.display = 'flex';
        state.selectedConstValues = []; // Start with all selected (empty means all)
        populateConstValueOptions();
        updateConstValueDisplay();
    } else {
        elements.navConstValueGroup.style.display = 'none';
    }
}

function populateConstValueOptions() {
    elements.constValueOptions.innerHTML = '';
    state.heatmapConstValues.forEach(val => {
        const option = document.createElement('div');
        option.className = 'multi-select-option';
        option.dataset.value = val;
        option.innerHTML = `<input type="checkbox" checked> <span>${val}</span>`;

        option.addEventListener('click', (e) => {
            e.stopPropagation();
            const checkbox = option.querySelector('input[type="checkbox"]');
            if (e.target !== checkbox) {
                checkbox.checked = !checkbox.checked;
            }
            toggleConstValue(val, checkbox.checked);
        });

        elements.constValueOptions.appendChild(option);
    });
}

function toggleConstValueDropdown(e) {
    e.stopPropagation();
    const isVisible = elements.constValueDropdown.style.display === 'block';
    elements.constValueDropdown.style.display = isVisible ? 'none' : 'block';
}

function toggleConstValue(val, isSelected) {
    if (isSelected) {
        // Remove from selected (empty array means all, so we need to track deselected)
        state.selectedConstValues = state.selectedConstValues.filter(v => v !== val);
    } else {
        // Add to selected (deselected list)
        if (!state.selectedConstValues.includes(val)) {
            state.selectedConstValues.push(val);
        }
    }
    updateConstValueDisplay();
    filterHeatmaps();
}

function selectAllConstValues() {
    state.selectedConstValues = [];
    elements.constValueOptions.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.checked = true;
    });
    elements.constValueOptions.querySelectorAll('.multi-select-option').forEach(opt => {
        opt.classList.remove('selected');
    });
    updateConstValueDisplay();
    filterHeatmaps();
}

function clearAllConstValues() {
    state.selectedConstValues = [...state.heatmapConstValues];
    elements.constValueOptions.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    updateConstValueDisplay();
    filterHeatmaps();
}

function updateConstValueDisplay() {
    const total = state.heatmapConstValues.length;
    const deselectedCount = state.selectedConstValues.length;
    const selectedCount = total - deselectedCount;

    if (selectedCount === 0) {
        elements.constValueDisplay.textContent = 'None selected';
    } else if (selectedCount === total) {
        elements.constValueDisplay.textContent = 'All Values';
    } else if (selectedCount <= 2) {
        // Show the actual selected values
        const selected = state.heatmapConstValues.filter(v => !state.selectedConstValues.includes(v));
        elements.constValueDisplay.textContent = selected.join(', ');
    } else {
        elements.constValueDisplay.textContent = `${selectedCount} selected`;
    }
}

function filterHeatmaps() {
    const selectedMetric = elements.navMetric.value;

    // Filter heatmaps based on selections
    state.filteredHeatmaps = state.heatmaps.filter(hm => {
        let matchesMetric = true;
        let matchesConstValue = true;

        // Use metadata if available
        const hmMetric = hm._metadata ? hm._metadata.metric : null;
        const hmConstValue = hm._metadata ? hm._metadata.const_value : null;

        // Filter by metric
        if (selectedMetric) {
            matchesMetric = hmMetric === selectedMetric;
        }

        // Filter by const values (multi-select)
        // state.selectedConstValues contains DESELECTED values
        // So we filter OUT heatmaps whose const_value is in the deselected list
        if (state.selectedConstValues.length > 0 && hmConstValue) {
            matchesConstValue = !state.selectedConstValues.includes(hmConstValue);
        }

        return matchesMetric && matchesConstValue;
    });

    // Reset index and display
    state.heatmapIndex = 0;
    if (state.filteredHeatmaps.length > 0) {
        renderHeatmap(state.filteredHeatmaps[0]);
    } else {
        document.getElementById('heatmap-container').innerHTML = '<p>No heatmaps match the selected filters.</p>';
    }
    updateHeatmapCounter();
}

function renderHeatmap(chartData) {
    const container = document.getElementById('heatmap-container');
    if (!chartData || !chartData.data) {
        container.innerHTML = '<p>No heatmap data</p>';
        return;
    }

    const colors = getChartColors();
    const config = { responsive: true, staticPlot: false, displayModeBar: true };
    const originalLayout = chartData.layout || {};

    // Get container width for responsive sizing - use full width with minimal padding
    const containerWidth = container.offsetWidth || 1200;

    // Calculate height based on data dimensions
    let dynamicHeight = originalLayout.height || 550;
    if (chartData.data[0] && chartData.data[0].z) {
        const numRows = chartData.data[0].z.length;
        dynamicHeight = Math.max(450, numRows * 45 + 120);
    }

    // Full width layout with theme colors - minimal margins
    const layout = {
        ...originalLayout,
        paper_bgcolor: colors.paper_bgcolor,
        plot_bgcolor: colors.plot_bgcolor,
        font: { color: colors.fontcolor, size: 12 },
        title: {
            ...originalLayout.title,
            font: { color: colors.fontcolor, size: 16 }
        },
        xaxis: {
            ...originalLayout.xaxis,
            gridcolor: colors.gridcolor,
            tickfont: { color: colors.fontcolor, size: 11 },
            titlefont: { color: colors.fontcolor, size: 14 }
        },
        yaxis: {
            ...originalLayout.yaxis,
            gridcolor: colors.gridcolor,
            tickfont: { color: colors.fontcolor, size: 11 },
            titlefont: { color: colors.fontcolor, size: 14 }
        },
        autosize: true,
        width: containerWidth - 10,  // Minimal padding
        height: dynamicHeight,
        margin: { l: 80, r: 100, t: 60, b: 60 }  // Reduced margins
    };

    // Clone data and update colorbar
    const data = JSON.parse(JSON.stringify(chartData.data));
    data.forEach(trace => {
        if (trace.type === 'heatmap') {
            // Ensure colorbar is styled for theme
            if (trace.colorbar) {
                trace.colorbar.tickfont = { color: colors.fontcolor, size: 11 };
                if (trace.colorbar.title) {
                    trace.colorbar.title.font = { color: colors.fontcolor, size: 12 };
                }
                trace.colorbar.thickness = 20;
                trace.colorbar.len = 0.9;
            }
            // Ensure text on cells is black for visibility
            if (trace.textfont) {
                trace.textfont.color = 'black';
            }
        }
    });

    // Keep annotation text visible on colored cells
    if (originalLayout.annotations) {
        layout.annotations = originalLayout.annotations.map(ann => ({
            ...ann,
            font: { ...ann.font, color: '#000000' }
        }));
    }

    Plotly.newPlot(container, data, layout, config);
}

function navigateHeatmap(direction) {
    const newIndex = state.heatmapIndex + direction;
    if (newIndex < 0 || newIndex >= state.filteredHeatmaps.length) return;
    state.heatmapIndex = newIndex;
    renderHeatmap(state.filteredHeatmaps[newIndex]);
    updateHeatmapCounter();
}

function updateHeatmapCounter() {
    const total = state.filteredHeatmaps.length;
    const current = total > 0 ? state.heatmapIndex + 1 : 0;
    document.getElementById('heatmap-counter').textContent = `${current} / ${total}`;
}

// ==================== Steps 3-7 ====================

function renderChart(container, chartData, fullWidth = false) {
    if (!chartData || !chartData.data) {
        container.innerHTML = '<p>No chart data</p>';
        return;
    }

    const colors = getChartColors();
    const isHeatmap = chartData.data.some(t => t.type === 'heatmap');
    const config = { responsive: true, displayModeBar: !isHeatmap, staticPlot: false };
    const originalLayout = chartData.layout || {};

    // Get container width for responsive sizing - try multiple sources
    let containerWidth = container.offsetWidth;
    if (!containerWidth || containerWidth < 100) {
        containerWidth = container.parentElement?.offsetWidth || 0;
    }
    if (!containerWidth || containerWidth < 100) {
        // Try to get width from cluster-section parent
        const clusterSection = container.closest('.cluster-section');
        if (clusterSection) {
            containerWidth = clusterSection.offsetWidth;
        }
    }
    if (!containerWidth || containerWidth < 100) {
        // Fallback to step panel width
        const stepPanel = container.closest('.step-panel');
        if (stepPanel) {
            containerWidth = stepPanel.offsetWidth - 60; // Account for padding
        }
    }
    if (!containerWidth || containerWidth < 100) {
        containerWidth = 1200; // Final fallback
    }

    const layout = {
        ...originalLayout,
        paper_bgcolor: colors.paper_bgcolor,
        plot_bgcolor: colors.plot_bgcolor,
        font: { color: colors.fontcolor },
        title: { ...originalLayout.title, font: { color: colors.fontcolor } },
        xaxis: { ...originalLayout.xaxis, gridcolor: colors.gridcolor, tickfont: { color: colors.fontcolor }, titlefont: { color: colors.fontcolor } },
        yaxis: { ...originalLayout.yaxis, gridcolor: colors.gridcolor, tickfont: { color: colors.fontcolor }, titlefont: { color: colors.fontcolor } },
        legend: { ...originalLayout.legend, font: { color: colors.fontcolor } },
        autosize: true
    };

    // For heatmaps in cluster sections, make them full width
    if (isHeatmap && (fullWidth || container.classList.contains('cluster-heatmap'))) {
        layout.width = containerWidth - 40;
        layout.autosize = true;
        layout.margin = { l: 80, r: 120, t: 60, b: 60 };
    }

    const data = JSON.parse(JSON.stringify(chartData.data));
    data.forEach(trace => {
        if (trace.type === 'heatmap' && trace.colorbar) {
            trace.colorbar.tickfont = { color: colors.fontcolor };
            if (trace.colorbar.title) trace.colorbar.title.font = { color: colors.fontcolor };
        }
    });

    if (originalLayout.annotations) {
        layout.annotations = originalLayout.annotations.map(ann => ({
            ...ann,
            font: { ...ann.font, color: '#000000' }
        }));
    }

    Plotly.newPlot(container, data, layout, config);
}

async function runStep3() {
    try {
        showLoading('Running PCA...');
        const result = await apiCall('/steps/pca', 'POST', { session_id: state.sessionId });
        renderChart(document.getElementById('pca-variance-chart'), result.pca_variance_chart);
        renderChart(document.getElementById('pca-scatter-chart'), result.pca_scatter);
        elements.step3Results.style.display = 'block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function runStep4() {
    try {
        showLoading('Running K-Means...');
        const kValue = elements.kmeansK.value ? parseInt(elements.kmeansK.value) : null;
        const result = await apiCall('/steps/kmeans', 'POST', { session_id: state.sessionId, k: kValue });

        document.getElementById('step4-k-used').textContent = result.k_used;
        document.getElementById('step4-filtered-count').textContent = result.num_variants_in_best_kmeans;
        renderChart(document.getElementById('kmeans-scatter-chart'), result.kmeans_scatter);

        // Store all cluster stats for metric switching
        state.kmeansAllStats = result.all_cluster_stats || {};

        // Render initial stats table (sharpe_ratio by default)
        renderKMeansStatsTable('sharpe_ratio');

        // Setup metric dropdown listener
        if (elements.kmeansStatsMetric) {
            elements.kmeansStatsMetric.value = 'sharpe_ratio';
            elements.kmeansStatsMetric.onchange = () => {
                renderKMeansStatsTable(elements.kmeansStatsMetric.value);
            };
        }

        elements.step4Results.style.display = 'block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function renderKMeansStatsTable(metric) {
    const stats = state.kmeansAllStats[metric] || [];
    const metricDisplayNames = {
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'profit_factor': 'Profit Factor'
    };
    const metricName = metricDisplayNames[metric] || metric;
    renderStatsTable('kmeans-stats-table', stats, metricName);
}

async function runStep5() {
    try {
        showLoading('Running HDBSCAN grid...');
        const minSizes = elements.hdbscanMinSizes.value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v));
        const minSamples = elements.hdbscanMinSamples.value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v));

        const request = {
            session_id: state.sessionId,
            grid_config: {
                min_cluster_sizes: minSizes,
                min_sample_sizes: minSamples,
                threshold_cluster_prob: parseFloat(elements.hdbscanThreshold.value)
            },
            ranking_metric: elements.rankingMetric.value
        };

        const result = await apiCall('/steps/hdbscan-grid', 'POST', request);
        state.availableConfigs = result.available_configs;

        renderChart(document.getElementById('hdbscan-grid-chart'), result.hdbscan_grid_chart);
        renderChart(document.getElementById('hdbscan-core-grid-chart'), result.hdbscan_core_grid_chart);
        renderConfigResults(result.config_results);
        populateStep6Dropdowns(result.available_configs);
        elements.step5Results.style.display = 'block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function runStep6() {
    try {
        showLoading('Running final HDBSCAN...');
        const request = {
            session_id: state.sessionId,
            min_cluster_size: parseInt(elements.finalMinClusterSize.value),
            min_samples: parseInt(elements.finalMinSamples.value),
            ranking_metric: elements.rankingMetric.value
        };

        const result = await apiCall('/steps/hdbscan-final', 'POST', request);

        // Store result for view toggling
        state.hdbscanFinalResult = result;

        document.getElementById('step6-num-clusters').textContent = result.num_clusters;
        document.getElementById('step6-core-count').textContent = result.num_core_points || '-';

        // Render all points view by default
        renderChart(document.getElementById('hdbscan-final-scatter-chart'), result.hdbscan_scatter);
        renderStatsTable('hdbscan-final-stats-table', result.hdbscan_cluster_stats);

        // Setup view toggle buttons
        setupHDBSCANViewToggle();

        elements.step6Results.style.display = 'block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function setupHDBSCANViewToggle() {
    const allPointsBtn = document.getElementById('hdbscan-all-points-btn');
    const corePointsBtn = document.getElementById('hdbscan-core-points-btn');
    const applyThresholdBtn = document.getElementById('apply-core-threshold-btn');

    if (allPointsBtn) {
        allPointsBtn.onclick = () => {
            allPointsBtn.classList.add('active');
            corePointsBtn.classList.remove('active');
            renderChart(document.getElementById('hdbscan-final-scatter-chart'), state.hdbscanFinalResult.hdbscan_scatter);
        };
    }

    if (corePointsBtn) {
        corePointsBtn.onclick = () => {
            corePointsBtn.classList.add('active');
            allPointsBtn.classList.remove('active');
            if (state.hdbscanFinalResult.hdbscan_core_scatter) {
                renderChart(document.getElementById('hdbscan-final-scatter-chart'), state.hdbscanFinalResult.hdbscan_core_scatter);
            }
        };
    }

    if (applyThresholdBtn) {
        applyThresholdBtn.onclick = async () => {
            // Re-run with new threshold (would need backend support)
            // For now, just toggle to core view
            corePointsBtn.click();
        };
    }
}

async function runStep7() {
    try {
        showLoading('Getting best clusters...');
        const request = {
            session_id: state.sessionId,
            num_best_clusters: parseInt(elements.numBestClusters.value),
            ranking_metric: elements.rankingMetric.value
        };

        const result = await apiCall('/steps/best-clusters', 'POST', request);
        document.getElementById('step7-cluster-ids').textContent = result.best_cluster_ids.join(', ') || 'None';
        renderBestClusters(result);
        elements.step7Results.style.display = 'block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function renderStatsTable(containerId, stats, metricName = null) {
    const container = document.getElementById(containerId);
    if (!stats || stats.length === 0) {
        container.innerHTML = '<p>No stats</p>';
        return;
    }

    // Column headers with metric name if provided
    const medianHeader = metricName ? `Median (${metricName})` : 'Median';
    const meanHeader = metricName ? `Mean (${metricName})` : 'Mean';
    const stdHeader = metricName ? `Std (${metricName})` : 'Std';

    let html = `<table><thead><tr><th>Cluster</th><th>Count</th><th>${medianHeader}</th><th>${meanHeader}</th><th>${stdHeader}</th></tr></thead><tbody>`;
    stats.forEach(s => {
        html += `<tr><td>${s.cluster_id}</td><td>${s.count}</td><td>${s.median?.toFixed(4) || '-'}</td><td>${s.mean?.toFixed(4) || '-'}</td><td>${s.std?.toFixed(4) || '-'}</td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

function renderConfigResults(configResults) {
    const container = document.getElementById('hdbscan-config-results');
    container.innerHTML = '';
    configResults.forEach(result => {
        const card = document.createElement('div');
        card.className = 'config-card';
        card.innerHTML = `<h5>min_size=${result.min_cluster_size}, min_samples=${result.min_samples}</h5><p>Clusters: ${result.num_clusters}</p>`;
        card.addEventListener('click', () => {
            document.querySelectorAll('.config-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            elements.finalMinClusterSize.value = result.min_cluster_size;
            elements.finalMinSamples.value = result.min_samples;
        });
        container.appendChild(card);
    });
}

function populateStep6Dropdowns(configs) {
    const minSizes = [...new Set(configs.map(c => c[0]))].sort((a, b) => a - b);
    const minSamples = [...new Set(configs.map(c => c[1]))].sort((a, b) => a - b);
    elements.finalMinClusterSize.innerHTML = minSizes.map(v => `<option value="${v}">${v}</option>`).join('');
    elements.finalMinSamples.innerHTML = minSamples.map(v => `<option value="${v}">${v}</option>`).join('');
    if (configs.length > 0) {
        elements.finalMinClusterSize.value = configs[0][0];
        elements.finalMinSamples.value = configs[0][1];
    }
}

function renderBestClusters(result) {
    const container = document.getElementById('best-clusters-container');
    container.innerHTML = '';

    result.best_cluster_ids.forEach((clusterId, idx) => {
        const section = document.createElement('div');
        section.className = 'cluster-section';
        const heatmaps = result.final_heatmaps[idx] || [];
        const coreHeatmaps = (result.final_core_heatmaps && result.final_core_heatmaps[idx]) || [];
        const clusterData = result.best_clusters_data[idx] || [];
        const constValues = (result.cluster_const_values && result.cluster_const_values[idx]) || [];
        const coreCount = (result.cluster_core_counts && result.cluster_core_counts[idx]) || 0;

        // Extract unique metrics from heatmap metadata
        const metrics = new Set();
        const constValuesFromHeatmaps = new Set();
        heatmaps.forEach(hm => {
            if (hm._metadata && hm._metadata.metric) {
                metrics.add(hm._metadata.metric);
            }
            if (hm._metadata && hm._metadata.const_value) {
                constValuesFromHeatmaps.add(hm._metadata.const_value);
            }
        });
        const metricsList = Array.from(metrics);
        const constValuesList = Array.from(constValuesFromHeatmaps).sort((a, b) => {
            const numA = parseFloat(a);
            const numB = parseFloat(b);
            if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
            return String(a).localeCompare(String(b));
        });

        // Build metric dropdown options
        let metricOptions = '<option value="">All Metrics</option>';
        metricsList.forEach(m => {
            metricOptions += `<option value="${m}">${m}</option>`;
        });

        // Build const value dropdown options
        let constValueOptions = '<option value="">All Values</option>';
        constValuesList.forEach(v => {
            constValueOptions += `<option value="${v}">${v}</option>`;
        });

        // Check if we have const values to show
        const showConstFilter = constValuesList.length > 0;

        // Check if we have core heatmaps available
        const hasCoreHeatmaps = coreHeatmaps.length > 0;

        section.innerHTML = `
            <h4>Cluster ${clusterId}</h4>
            <div class="cluster-info">
                <span class="info-badge">${clusterData.length} variants</span>
                <span class="info-badge">${coreCount} core points</span>
                <span class="info-badge">${heatmaps.length} heatmaps</span>
                ${constValues.length > 0 ? `<span class="info-badge">Const values: ${constValues.join(', ')}</span>` : ''}
            </div>
            <div class="cluster-heatmap-controls">
                <div class="cluster-filter-row">
                    ${hasCoreHeatmaps ? `
                    <div class="cluster-view-toggle">
                        <label>Highlight:</label>
                        <div class="view-toggle-buttons">
                            <button class="btn btn-small view-toggle-btn cluster-all-btn active" data-cluster="${idx}">All Variants</button>
                            <button class="btn btn-small view-toggle-btn cluster-core-btn" data-cluster="${idx}">Core Only</button>
                        </div>
                    </div>
                    ` : ''}
                    <div class="cluster-metric-filter">
                        <label>Metric:</label>
                        <select class="cluster-metric-select" data-cluster="${idx}">
                            ${metricOptions}
                        </select>
                    </div>
                    ${showConstFilter ? `
                    <div class="cluster-const-filter">
                        <label>Const Value:</label>
                        <select class="cluster-const-select" data-cluster="${idx}">
                            ${constValueOptions}
                        </select>
                    </div>
                    ` : ''}
                </div>
                <div class="cluster-heatmap-nav">
                    <button class="btn btn-small prev-cluster-heatmap" data-cluster="${idx}">&#9664; Prev</button>
                    <span id="cluster-${idx}-counter">1 / ${heatmaps.length}</span>
                    <button class="btn btn-small next-cluster-heatmap" data-cluster="${idx}">Next &#9654;</button>
                </div>
            </div>
            <div class="cluster-heatmap" id="cluster-${idx}-heatmap"></div>
            <h5>Cluster Variants</h5>
            <div class="cluster-table data-table-container" id="cluster-${idx}-table"></div>
        `;
        container.appendChild(section);

        // Store heatmaps and filtered view
        section._allHeatmaps = heatmaps;
        section._allCoreHeatmaps = coreHeatmaps;
        section._filteredHeatmaps = heatmaps;
        section._filteredCoreHeatmaps = coreHeatmaps;
        section._heatmapIndex = 0;
        section._showCoreOnly = false;

        if (heatmaps.length > 0) {
            renderChart(document.getElementById(`cluster-${idx}-heatmap`), heatmaps[0]);
        } else {
            document.getElementById(`cluster-${idx}-heatmap`).innerHTML = '<p class="no-data">No heatmaps available for this cluster</p>';
        }
        if (clusterData.length > 0) {
            renderDataTable(`cluster-${idx}-table`, clusterData.slice(0, 50));
        }
    });

    // Combined filter function for both metric and const value
    function applyClusterFilters(section, idx) {
        const metricSelect = section.querySelector('.cluster-metric-select');
        const constSelect = section.querySelector('.cluster-const-select');

        const selectedMetric = metricSelect ? metricSelect.value : '';
        const selectedConstValue = constSelect ? constSelect.value : '';

        // Filter both all heatmaps and core heatmaps by the same criteria
        const filterFn = (hm) => {
            let matchesMetric = true;
            let matchesConstValue = true;

            if (selectedMetric && hm._metadata) {
                matchesMetric = hm._metadata.metric === selectedMetric;
            }
            if (selectedConstValue && hm._metadata) {
                matchesConstValue = hm._metadata.const_value === selectedConstValue;
            }

            return matchesMetric && matchesConstValue;
        };

        section._filteredHeatmaps = section._allHeatmaps.filter(filterFn);
        section._filteredCoreHeatmaps = section._allCoreHeatmaps.filter(filterFn);

        // Reset index and update display
        section._heatmapIndex = 0;

        // Get current heatmaps based on view mode
        const currentHeatmaps = section._showCoreOnly ? section._filteredCoreHeatmaps : section._filteredHeatmaps;
        const total = currentHeatmaps.length;
        document.getElementById(`cluster-${idx}-counter`).textContent = `${total > 0 ? 1 : 0} / ${total}`;

        if (total > 0) {
            renderChart(document.getElementById(`cluster-${idx}-heatmap`), currentHeatmaps[0]);
        } else {
            document.getElementById(`cluster-${idx}-heatmap`).innerHTML = '<p class="no-data">No heatmaps match the selected filters</p>';
        }
    }

    // View toggle handlers (All Variants vs Core Only)
    container.querySelectorAll('.cluster-all-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const idx = parseInt(this.dataset.cluster);
            const section = this.closest('.cluster-section');

            // Update toggle state
            section._showCoreOnly = false;
            section.querySelector('.cluster-all-btn').classList.add('active');
            section.querySelector('.cluster-core-btn').classList.remove('active');

            // Reset to filtered all heatmaps
            section._heatmapIndex = 0;
            const total = section._filteredHeatmaps.length;
            document.getElementById(`cluster-${idx}-counter`).textContent = `${total > 0 ? 1 : 0} / ${total}`;

            if (total > 0) {
                renderChart(document.getElementById(`cluster-${idx}-heatmap`), section._filteredHeatmaps[0]);
            } else {
                document.getElementById(`cluster-${idx}-heatmap`).innerHTML = '<p class="no-data">No heatmaps available</p>';
            }
        });
    });

    container.querySelectorAll('.cluster-core-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const idx = parseInt(this.dataset.cluster);
            const section = this.closest('.cluster-section');

            // Update toggle state
            section._showCoreOnly = true;
            section.querySelector('.cluster-core-btn').classList.add('active');
            section.querySelector('.cluster-all-btn').classList.remove('active');

            // Reset to filtered core heatmaps
            section._heatmapIndex = 0;
            const total = section._filteredCoreHeatmaps.length;
            document.getElementById(`cluster-${idx}-counter`).textContent = `${total > 0 ? 1 : 0} / ${total}`;

            if (total > 0) {
                renderChart(document.getElementById(`cluster-${idx}-heatmap`), section._filteredCoreHeatmaps[0]);
            } else {
                document.getElementById(`cluster-${idx}-heatmap`).innerHTML = '<p class="no-data">No core heatmaps available</p>';
            }
        });
    });

    // Metric filter handlers
    container.querySelectorAll('.cluster-metric-select').forEach(select => {
        select.addEventListener('change', function() {
            const idx = parseInt(this.dataset.cluster);
            const section = this.closest('.cluster-section');
            applyClusterFilters(section, idx);
        });
    });

    // Const value filter handlers
    container.querySelectorAll('.cluster-const-select').forEach(select => {
        select.addEventListener('change', function() {
            const idx = parseInt(this.dataset.cluster);
            const section = this.closest('.cluster-section');
            applyClusterFilters(section, idx);
        });
    });

    // Navigation handlers
    container.querySelectorAll('.prev-cluster-heatmap').forEach(btn => {
        btn.addEventListener('click', function() {
            const idx = parseInt(this.dataset.cluster);
            const section = this.closest('.cluster-section');
            const currentHeatmaps = section._showCoreOnly ? section._filteredCoreHeatmaps : section._filteredHeatmaps;

            if (section._heatmapIndex > 0) {
                section._heatmapIndex--;
                renderChart(document.getElementById(`cluster-${idx}-heatmap`), currentHeatmaps[section._heatmapIndex]);
                document.getElementById(`cluster-${idx}-counter`).textContent = `${section._heatmapIndex + 1} / ${currentHeatmaps.length}`;
            }
        });
    });
    container.querySelectorAll('.next-cluster-heatmap').forEach(btn => {
        btn.addEventListener('click', function() {
            const idx = parseInt(this.dataset.cluster);
            const section = this.closest('.cluster-section');
            const currentHeatmaps = section._showCoreOnly ? section._filteredCoreHeatmaps : section._filteredHeatmaps;

            if (section._heatmapIndex < currentHeatmaps.length - 1) {
                section._heatmapIndex++;
                renderChart(document.getElementById(`cluster-${idx}-heatmap`), currentHeatmaps[section._heatmapIndex]);
                document.getElementById(`cluster-${idx}-counter`).textContent = `${section._heatmapIndex + 1} / ${currentHeatmaps.length}`;
            }
        });
    });
}
