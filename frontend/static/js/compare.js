/**
 * Compare Variants - Frontend JavaScript
 *
 * State management and UI logic for comparing multiple variant CSVs.
 */

(function() {
    'use strict';

    // Application state
    const state = {
        sessionId: null,
        selectedFiles: [],
        variants: [],
        columns: [],
        variantMetrics: [],
        aggregateStats: {},
        currentChart: 'table',
        aliasMap: {},  // Maps original variant name to alias
        visibleVariants: {},  // Maps variant name to visibility for charts (true/false)
        tableVisibleVariants: {},  // Maps variant name to visibility for table (true/false)
        browser: {
            currentPath: '',
            selectedFiles: []
        }
    };

    // Variant colors (must match backend VARIANT_COLORS)
    const VARIANT_COLORS = [
        '#1f77b4',  // Blue
        '#ff7f0e',  // Orange
        '#2ca02c',  // Green
        '#d62728',  // Red
        '#9467bd',  // Purple
        '#8c564b',  // Brown
        '#e377c2',  // Pink
        '#7f7f7f',  // Gray
        '#bcbd22',  // Olive
        '#17becf',  // Cyan
    ];

    // DOM element cache
    const elements = {};

    /**
     * Initialize the application
     */
    function init() {
        cacheElements();
        setupEventListeners();
        browsePath('');  // Start with home directory
    }

    /**
     * Cache DOM elements for performance
     */
    function cacheElements() {
        elements.browserPath = document.getElementById('browser-path');
        elements.browserContent = document.getElementById('browser-content');
        elements.browseUpBtn = document.getElementById('browse-up-btn');
        elements.browseHomeBtn = document.getElementById('browse-home-btn');
        elements.selectedCount = document.getElementById('selected-count');
        elements.selectedList = document.getElementById('selected-list');
        elements.clearSelectionBtn = document.getElementById('clear-selection-btn');
        elements.loadFilesBtn = document.getElementById('load-files-btn');
        elements.aliasInputs = document.getElementById('alias-inputs');
        elements.thresholdX = document.getElementById('threshold-x');
        elements.thresholdY = document.getElementById('threshold-y');
        elements.calculateMetricsBtn = document.getElementById('calculate-metrics-btn');
        elements.cumulativeChartContainer = document.getElementById('cumulative-chart-container');
        elements.commonDateRangeInfo = document.getElementById('common-date-range-info');
        elements.commonDateRangeText = document.getElementById('common-date-range-text');
        elements.variantToggles = document.getElementById('variant-toggles');
        elements.dateFrom = document.getElementById('date-from');
        elements.dateTo = document.getElementById('date-to');
        elements.applyDateFilterBtn = document.getElementById('apply-date-filter-btn');
        elements.clearDateFilterBtn = document.getElementById('clear-date-filter-btn');
        elements.chartTabs = document.getElementById('chart-tabs');
        elements.chartContainer = document.getElementById('chart-container');
        elements.exportCsvBtn = document.getElementById('export-csv-btn');
        elements.exportPdfBtn = document.getElementById('export-pdf-btn');
        // Export modal elements
        elements.exportModal = document.getElementById('export-modal');
        elements.exportModalTitle = document.getElementById('export-modal-title');
        elements.exportTitle = document.getElementById('export-title');
        elements.exportFilename = document.getElementById('export-filename');
        elements.exportExtensionHint = document.getElementById('export-extension-hint');
        elements.modalCloseBtn = document.getElementById('modal-close-btn');
        elements.modalCancelBtn = document.getElementById('modal-cancel-btn');
        elements.modalExportBtn = document.getElementById('modal-export-btn');
        elements.step1 = document.getElementById('step-1');
        elements.step2 = document.getElementById('step-2');
        elements.step3 = document.getElementById('step-3');
        elements.step4 = document.getElementById('step-4');
        elements.tableVariantToggles = document.getElementById('table-variant-toggles');
    }

    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        elements.browseUpBtn.addEventListener('click', () => {
            if (state.browser.parentPath) {
                browsePath(state.browser.parentPath);
            }
        });

        elements.browseHomeBtn.addEventListener('click', () => {
            browsePath('');
        });

        elements.clearSelectionBtn.addEventListener('click', clearSelection);
        elements.loadFilesBtn.addEventListener('click', loadFiles);
        elements.calculateMetricsBtn.addEventListener('click', calculateMetrics);

        // Date filter buttons
        elements.applyDateFilterBtn.addEventListener('click', applyDateFilter);
        elements.clearDateFilterBtn.addEventListener('click', clearDateFilter);

        // Chart tab clicks
        elements.chartTabs.addEventListener('click', (e) => {
            if (e.target.classList.contains('chart-tab')) {
                const chartType = e.target.dataset.chart;
                setActiveTab(chartType);
                generateChart(chartType);
            }
        });

        // Export buttons - both use modal now
        if (elements.exportCsvBtn) {
            elements.exportCsvBtn.addEventListener('click', () => showExportModal('csv'));
        }
        if (elements.exportPdfBtn) {
            elements.exportPdfBtn.addEventListener('click', () => showExportModal('pdf'));
        }

        // Export modal buttons - get fresh references in case they weren't cached
        const modalCloseBtn = elements.modalCloseBtn || document.getElementById('modal-close-btn');
        const modalCancelBtn = elements.modalCancelBtn || document.getElementById('modal-cancel-btn');
        const modalExportBtn = elements.modalExportBtn || document.getElementById('modal-export-btn');
        const exportModal = elements.exportModal || document.getElementById('export-modal');

        if (modalCloseBtn) {
            modalCloseBtn.addEventListener('click', () => hideExportModal());
        }
        if (modalCancelBtn) {
            modalCancelBtn.addEventListener('click', () => hideExportModal());
        }
        if (modalExportBtn) {
            modalExportBtn.addEventListener('click', () => confirmExport());
        }

        // Close modal on overlay click
        if (exportModal) {
            exportModal.addEventListener('click', (e) => {
                if (e.target === exportModal) {
                    hideExportModal();
                }
            });
        }

        // Window resize handler for responsive charts
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                // Relayout cumulative chart to fill container
                if (elements.cumulativeChartContainer && elements.cumulativeChartContainer.data) {
                    const containerHeight = elements.cumulativeChartContainer.clientHeight;
                    const containerWidth = elements.cumulativeChartContainer.clientWidth;
                    Plotly.relayout(elements.cumulativeChartContainer, {
                        height: containerHeight > 0 ? containerHeight : Math.floor(window.innerHeight * 0.75),
                        width: containerWidth > 0 ? containerWidth : undefined
                    });
                }
                // Relayout other charts
                if (elements.chartContainer && elements.chartContainer.data) {
                    Plotly.Plots.resize(elements.chartContainer);
                }
            }, 250);
        });
    }

    /**
     * Browse a directory path
     */
    async function browsePath(path) {
        elements.browserContent.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

        try {
            const response = await fetch(`/compare/browse?path=${encodeURIComponent(path)}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to browse directory');
            }

            const data = await response.json();
            state.browser.currentPath = data.current_path;
            state.browser.parentPath = data.parent_path;

            renderBrowser(data);
        } catch (error) {
            elements.browserContent.innerHTML = `<div class="empty-state"><p>Error: ${error.message}</p></div>`;
        }
    }

    /**
     * Render the file browser
     */
    function renderBrowser(data) {
        elements.browserPath.textContent = data.current_path;
        elements.browseUpBtn.disabled = !data.parent_path;

        let html = '';

        // Directories first
        for (const dir of data.directories) {
            html += `
                <div class="browser-item browser-dir" data-path="${escapeHtml(dir.path)}">
                    <span class="browser-item-icon">&#128193;</span>
                    <span class="browser-item-name">${escapeHtml(dir.name)}</span>
                </div>
            `;
        }

        // Then files
        for (const file of data.files) {
            const isSelected = state.selectedFiles.includes(file.path);
            html += `
                <div class="browser-item browser-file ${isSelected ? 'selected' : ''}" data-path="${escapeHtml(file.path)}">
                    <span class="browser-item-icon">&#128196;</span>
                    <span class="browser-item-name">${escapeHtml(file.name)}</span>
                    <span class="browser-item-size">${file.size_display}</span>
                </div>
            `;
        }

        if (!data.directories.length && !data.files.length) {
            html = '<div class="empty-state"><p>No CSV files found in this directory</p></div>';
        }

        elements.browserContent.innerHTML = html;

        // Add click handlers
        elements.browserContent.querySelectorAll('.browser-dir').forEach(el => {
            el.addEventListener('click', () => browsePath(el.dataset.path));
        });

        elements.browserContent.querySelectorAll('.browser-file').forEach(el => {
            el.addEventListener('click', () => toggleFileSelection(el.dataset.path));
        });
    }

    /**
     * Toggle file selection
     */
    function toggleFileSelection(path) {
        const index = state.selectedFiles.indexOf(path);
        if (index === -1) {
            state.selectedFiles.push(path);
        } else {
            state.selectedFiles.splice(index, 1);
        }
        updateSelectionUI();
        // Re-render to update selected state
        browsePath(state.browser.currentPath);
    }

    /**
     * Update the selection UI
     */
    function updateSelectionUI() {
        elements.selectedCount.textContent = state.selectedFiles.length;
        elements.loadFilesBtn.disabled = state.selectedFiles.length === 0;

        let html = '';
        for (const path of state.selectedFiles) {
            const name = path.split('/').pop();
            html += `
                <span class="selected-file-tag">
                    ${escapeHtml(name)}
                    <span class="remove-file-btn" data-path="${escapeHtml(path)}">&times;</span>
                </span>
            `;
        }
        elements.selectedList.innerHTML = html;

        // Add remove handlers
        elements.selectedList.querySelectorAll('.remove-file-btn').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                const path = el.dataset.path;
                state.selectedFiles = state.selectedFiles.filter(p => p !== path);
                updateSelectionUI();
                browsePath(state.browser.currentPath);
            });
        });
    }

    /**
     * Clear all selected files
     */
    function clearSelection() {
        state.selectedFiles = [];
        updateSelectionUI();
        browsePath(state.browser.currentPath);
    }

    /**
     * Load selected files
     */
    async function loadFiles() {
        if (state.selectedFiles.length === 0) return;

        elements.loadFilesBtn.disabled = true;
        elements.loadFilesBtn.textContent = 'Loading...';

        try {
            const response = await fetch('/compare/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ csv_paths: state.selectedFiles })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to load files');
            }

            const data = await response.json();
            state.sessionId = data.session_id;
            state.variants = data.variants;
            state.columns = data.columns;

            // Initialize alias map with original names and render alias inputs
            state.aliasMap = {};
            state.variants.forEach(v => {
                state.aliasMap[v.name] = v.name;  // Default alias is the original name
            });
            renderAliasInputs();

            // Enable step 2
            elements.step2.classList.remove('step-disabled');
            elements.step1.querySelector('h3').classList.add('step-complete');

            // Scroll to step 2
            elements.step2.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            alert('Error loading files: ' + error.message);
        } finally {
            elements.loadFilesBtn.disabled = false;
            elements.loadFilesBtn.textContent = 'Load Selected Files';
        }
    }

    // Fixed column names - no user selection needed
    const FIXED_COLUMNS = {
        pnl: 'Pnl%',
        year: 'Year',
        trades: 'Trades',
        expiry: 'Expiry',
        minNotional: 'Notional min pnl'
    };

    /**
     * Render alias input fields for each variant
     */
    function renderAliasInputs() {
        if (!state.variants || state.variants.length === 0) {
            elements.aliasInputs.innerHTML = '<p class="form-hint">No variants loaded</p>';
            return;
        }

        elements.aliasInputs.innerHTML = state.variants.map(v => `
            <div class="alias-input-group">
                <label title="${escapeHtml(v.name)}">${escapeHtml(v.name)}</label>
                <input type="text"
                       class="alias-input"
                       data-original="${escapeHtml(v.name)}"
                       value="${escapeHtml(state.aliasMap[v.name] || v.name)}"
                       placeholder="Enter display name">
            </div>
        `).join('');

        // Add event listeners to update alias map on change
        elements.aliasInputs.querySelectorAll('.alias-input').forEach(input => {
            input.addEventListener('input', (e) => {
                const originalName = e.target.dataset.original;
                const aliasValue = e.target.value.trim();
                state.aliasMap[originalName] = aliasValue || originalName;
            });
        });
    }

    /**
     * Get display name for a variant (alias or original name)
     */
    function getDisplayName(originalName) {
        return state.aliasMap[originalName] || originalName;
    }

    /**
     * Render variant visibility toggle checkboxes
     */
    function renderVariantToggles() {
        if (!state.variants || state.variants.length === 0) {
            elements.variantToggles.style.display = 'none';
            return;
        }

        // Initialize visibility state for all variants (all visible by default)
        state.variants.forEach(v => {
            if (state.visibleVariants[v.name] === undefined) {
                state.visibleVariants[v.name] = true;
            }
        });

        let html = '<span class="variant-toggles-label">Show Variants:</span>';
        state.variants.forEach((v, i) => {
            const displayName = getDisplayName(v.name);
            const color = VARIANT_COLORS[i % VARIANT_COLORS.length];
            const isChecked = state.visibleVariants[v.name] !== false;
            html += `
                <div class="variant-toggle-item">
                    <input type="checkbox"
                           id="toggle-${i}"
                           data-variant="${escapeHtml(v.name)}"
                           ${isChecked ? 'checked' : ''}>
                    <span class="variant-color-dot" style="background-color: ${color};"></span>
                    <label for="toggle-${i}">${escapeHtml(displayName)}</label>
                </div>
            `;
        });

        elements.variantToggles.innerHTML = html;
        elements.variantToggles.style.display = 'flex';

        // Add event listeners for toggle changes
        elements.variantToggles.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const variantName = e.target.dataset.variant;
                state.visibleVariants[variantName] = e.target.checked;
                // Refresh the cumulative chart with updated visibility
                updateChartVisibility();
            });
        });
    }

    /**
     * Render variant visibility toggle checkboxes for table (Step 4)
     */
    function renderTableVariantToggles() {
        if (!state.variants || state.variants.length === 0) {
            elements.tableVariantToggles.style.display = 'none';
            return;
        }

        // Initialize visibility state for all variants (all visible by default)
        state.variants.forEach(v => {
            if (state.tableVisibleVariants[v.name] === undefined) {
                state.tableVisibleVariants[v.name] = true;
            }
        });

        let html = '<span class="variant-toggles-label">Show Variants:</span>';
        state.variants.forEach((v, i) => {
            const displayName = getDisplayName(v.name);
            const color = VARIANT_COLORS[i % VARIANT_COLORS.length];
            const isChecked = state.tableVisibleVariants[v.name] !== false;
            html += `
                <div class="variant-toggle-item">
                    <input type="checkbox"
                           id="table-toggle-${i}"
                           data-variant="${escapeHtml(v.name)}"
                           ${isChecked ? 'checked' : ''}>
                    <span class="variant-color-dot" style="background-color: ${color};"></span>
                    <label for="table-toggle-${i}">${escapeHtml(displayName)}</label>
                </div>
            `;
        });

        elements.tableVariantToggles.innerHTML = html;
        elements.tableVariantToggles.style.display = 'flex';

        // Add event listeners for toggle changes
        elements.tableVariantToggles.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const variantName = e.target.dataset.variant;
                state.tableVisibleVariants[variantName] = e.target.checked;
                // Refresh the current chart/table with updated visibility
                generateChart(state.currentChart);
            });
        });
    }

    /**
     * Get list of visible variants for table
     */
    function getVisibleTableVariants() {
        return state.variants.filter(v => state.tableVisibleVariants[v.name] !== false);
    }

    /**
     * Update chart trace visibility based on toggle state
     */
    function updateChartVisibility() {
        if (!elements.cumulativeChartContainer || !elements.cumulativeChartContainer.data) {
            return;
        }

        const traceUpdates = [];
        const traces = elements.cumulativeChartContainer.data;

        // Build visibility array for all traces
        traces.forEach((trace, index) => {
            // Find which variant this trace belongs to by checking the name
            let isVisible = true;
            for (const [variantName, visible] of Object.entries(state.visibleVariants)) {
                const displayName = getDisplayName(variantName);
                // Check if trace name matches variant (either cumulative or drawdown)
                if (trace.name === displayName || trace.name === `${displayName} (Drawdown)`) {
                    isVisible = visible;
                    break;
                }
            }
            traceUpdates.push(isVisible);
        });

        // Use Plotly.restyle to update visibility
        Plotly.restyle(elements.cumulativeChartContainer, { visible: traceUpdates });
    }

    /**
     * Calculate metrics for all variants
     */
    async function calculateMetrics() {
        if (!state.sessionId) return;

        elements.calculateMetricsBtn.disabled = true;
        elements.calculateMetricsBtn.textContent = 'Calculating...';

        try {
            const response = await fetch('/compare/calculate-metrics', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: state.sessionId,
                    pnl_column: FIXED_COLUMNS.pnl,
                    year_column: FIXED_COLUMNS.year,
                    trades_column: FIXED_COLUMNS.trades,
                    expiry_column: FIXED_COLUMNS.expiry,
                    min_notional_column: FIXED_COLUMNS.minNotional,
                    threshold_x_pct: parseFloat(elements.thresholdX.value) || -5.0,
                    threshold_y_pct: parseFloat(elements.thresholdY.value) || -10.0
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to calculate metrics');
            }

            const data = await response.json();
            state.variantMetrics = data.variant_metrics;
            state.aggregateStats = data.aggregate_stats;
            state.commonDateRange = data.common_date_range;

            // Show common date range info
            if (data.common_date_range && data.common_date_range.start && data.common_date_range.end) {
                elements.commonDateRangeText.textContent = `${data.common_date_range.start} to ${data.common_date_range.end}`;
                elements.commonDateRangeInfo.style.display = 'block';
            } else {
                elements.commonDateRangeInfo.style.display = 'none';
            }

            // Update UI
            elements.step2.querySelector('h3').classList.add('step-complete');
            elements.step3.classList.remove('step-disabled');
            elements.step4.classList.remove('step-disabled');

            // Initialize visibility for all variants and render toggles
            state.visibleVariants = {};
            state.tableVisibleVariants = {};
            state.variants.forEach(v => {
                state.visibleVariants[v.name] = true;
                state.tableVisibleVariants[v.name] = true;
            });
            renderVariantToggles();
            renderTableVariantToggles();

            // Generate performance chart in Step 3 (combined cumulative + drawdown)
            generateCumulativeChart();

            // Generate initial table view
            state.currentChart = 'table';
            generateChart('table');

            // Scroll to results
            elements.step3.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            alert('Error calculating metrics: ' + error.message);
        } finally {
            elements.calculateMetricsBtn.disabled = false;
            elements.calculateMetricsBtn.textContent = 'Calculate Metrics';
        }
    }

    /**
     * Apply date filter to performance chart
     */
    function applyDateFilter() {
        generateCumulativeChart();
    }

    /**
     * Clear date filter and refresh chart
     */
    function clearDateFilter() {
        elements.dateFrom.value = '';
        elements.dateTo.value = '';
        generateCumulativeChart();
    }

    /**
     * Generate cumulative PnL chart for Step 3
     */
    async function generateCumulativeChart() {
        if (!state.sessionId) return;

        elements.cumulativeChartContainer.innerHTML = '<div class="loading"><div class="spinner"></div>Loading chart...</div>';

        // Get date filter values
        const dateFrom = elements.dateFrom.value || null;
        const dateTo = elements.dateTo.value || null;

        try {
            const response = await fetch('/compare/generate-chart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: state.sessionId,
                    chart_type: 'line',
                    normalize: false,
                    date_from: dateFrom,
                    date_to: dateTo,
                    alias_map: state.aliasMap
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to generate cumulative chart');
            }

            const data = await response.json();

            if (data.chart_data && data.chart_data.data) {
                elements.cumulativeChartContainer.innerHTML = '';

                // Apply theme colors
                const isDark = document.body.classList.contains('dark-theme');
                const layout = data.chart_data.layout || {};

                // Set chart to fill container without overflow (account for padding)
                const containerRect = elements.cumulativeChartContainer.getBoundingClientRect();
                const padding = 20; // Account for container padding (10px each side)
                const chartHeight = Math.max(containerRect.height - padding, 400);
                const chartWidth = Math.max(containerRect.width - padding, 300);

                layout.height = chartHeight;
                layout.width = chartWidth;
                layout.autosize = false; // Disable autosize to respect our dimensions
                layout.margin = layout.margin || {};
                layout.margin.l = layout.margin.l || 60;
                layout.margin.r = layout.margin.r || 60;
                layout.margin.t = layout.margin.t || 50;
                layout.margin.b = layout.margin.b || 50;

                layout.paper_bgcolor = isDark ? '#1e293b' : '#ffffff';
                layout.plot_bgcolor = isDark ? '#0f172a' : '#f8fafc';
                layout.font = layout.font || {};
                layout.font.color = isDark ? '#f1f5f9' : '#1e293b';

                if (layout.xaxis) {
                    layout.xaxis.gridcolor = isDark ? '#334155' : '#e2e8f0';
                }
                if (layout.yaxis) {
                    layout.yaxis.gridcolor = isDark ? '#334155' : '#e2e8f0';
                }
                if (layout.yaxis2) {
                    layout.yaxis2.gridcolor = isDark ? '#334155' : '#e2e8f0';
                }

                // Apply visibility state to traces
                const chartData = data.chart_data.data.map(trace => {
                    let isVisible = true;
                    for (const [variantName, visible] of Object.entries(state.visibleVariants)) {
                        const displayName = getDisplayName(variantName);
                        if (trace.name === displayName || trace.name === `${displayName} (Drawdown)`) {
                            isVisible = visible;
                            break;
                        }
                    }
                    return { ...trace, visible: isVisible };
                });

                Plotly.newPlot(elements.cumulativeChartContainer, chartData, layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d']
                });
            } else {
                elements.cumulativeChartContainer.innerHTML = '<div class="empty-state"><p>No chart data available</p></div>';
            }

        } catch (error) {
            elements.cumulativeChartContainer.innerHTML = `<div class="empty-state"><p>Error: ${error.message}</p></div>`;
        }
    }

    /**
     * Set active chart tab
     */
    function setActiveTab(chartType) {
        elements.chartTabs.querySelectorAll('.chart-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.chart === chartType);
        });
        state.currentChart = chartType;
    }

    /**
     * Generate a chart
     */
    async function generateChart(chartType) {
        if (!state.sessionId || state.variantMetrics.length === 0) return;

        elements.chartContainer.innerHTML = '<div class="loading"><div class="spinner"></div>Generating chart...</div>';

        // Get selected variants for table/charts
        const visibleVariants = getVisibleTableVariants();
        const selectedVariantNames = visibleVariants.map(v => v.name);

        try {
            const response = await fetch('/compare/generate-chart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: state.sessionId,
                    chart_type: chartType,
                    normalize: chartType === 'radar',
                    alias_map: state.aliasMap,
                    selected_variants: selectedVariantNames
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to generate chart');
            }

            const data = await response.json();

            if (chartType === 'table') {
                renderComparisonTable(data.table_data);
            } else if (chartType === 'ranking_table') {
                renderRankingTable(data.table_data);
            } else {
                renderPlotlyChart(data.chart_data);
            }

        } catch (error) {
            elements.chartContainer.innerHTML = `<div class="empty-state"><p>Error: ${error.message}</p></div>`;
        }
    }

    /**
     * Show export modal for CSV or PDF
     */
    function showExportModal(format) {
        if (!state.sessionId || state.variantMetrics.length === 0) {
            alert('No data to export. Please calculate metrics first.');
            return;
        }

        // Store the format for use in confirmExport
        state.exportFormat = format;

        // Set default filename with timestamp
        const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');

        // Get fresh references if needed
        if (!elements.exportModal) {
            elements.exportModal = document.getElementById('export-modal');
        }
        if (!elements.exportModalTitle) {
            elements.exportModalTitle = document.getElementById('export-modal-title');
        }
        if (!elements.exportTitle) {
            elements.exportTitle = document.getElementById('export-title');
        }
        if (!elements.exportFilename) {
            elements.exportFilename = document.getElementById('export-filename');
        }
        if (!elements.exportExtensionHint) {
            elements.exportExtensionHint = document.getElementById('export-extension-hint');
        }
        if (!elements.modalExportBtn) {
            elements.modalExportBtn = document.getElementById('modal-export-btn');
        }

        // Update modal title and button based on format
        const formatUpper = format.toUpperCase();
        if (elements.exportModalTitle) {
            elements.exportModalTitle.textContent = `Export as ${formatUpper}`;
        }
        if (elements.exportExtensionHint) {
            elements.exportExtensionHint.textContent = `The .${format} extension will be added automatically`;
        }
        if (elements.modalExportBtn) {
            elements.modalExportBtn.textContent = `Export ${formatUpper}`;
        }

        if (elements.exportTitle) {
            elements.exportTitle.value = 'Variant Comparison Report';
        }
        if (elements.exportFilename) {
            elements.exportFilename.value = `compare_variants_${timestamp}`;
        }

        // Show modal
        if (elements.exportModal) {
            elements.exportModal.classList.add('show');
            if (elements.exportFilename) {
                elements.exportFilename.focus();
                elements.exportFilename.select();
            }
        } else {
            console.error('Could not find export modal');
        }
    }

    /**
     * Hide PDF export modal
     */
    function hideExportModal() {
        const modal = elements.exportModal || document.getElementById('export-modal');
        if (modal) {
            modal.classList.remove('show');
        }
    }

    /**
     * Confirm export with custom filename and title (handles both CSV and PDF)
     */
    async function confirmExport() {
        const format = state.exportFormat || 'pdf';
        const title = (elements.exportTitle ? elements.exportTitle.value.trim() : '') || 'Variant Comparison Report';
        const filename = elements.exportFilename.value.trim() || 'compare_variants';

        // Get selected variants for export
        const visibleVariants = getVisibleTableVariants();
        const selectedVariantNames = visibleVariants.map(v => v.name);

        if (selectedVariantNames.length === 0) {
            alert('No variants selected. Please select at least one variant to export.');
            return;
        }

        // Hide modal first
        hideExportModal();

        // Get the appropriate button
        const exportBtn = format === 'csv' ? elements.exportCsvBtn : elements.exportPdfBtn;
        const originalText = exportBtn.innerHTML;
        exportBtn.disabled = true;
        exportBtn.innerHTML = `<span class="icon">&#8987;</span> Exporting...`;

        try {
            const response = await fetch(`/compare/export-${format}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: state.sessionId,
                    alias_map: state.aliasMap,
                    format: format,
                    filename: filename,
                    title: title,
                    selected_variants: selectedVariantNames
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `Failed to export ${format.toUpperCase()}`);
            }

            // Get the blob and download it
            const blob = await response.blob();

            // Create download link with custom filename
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

        } catch (error) {
            alert(`Export failed: ${error.message}`);
        } finally {
            exportBtn.disabled = false;
            exportBtn.innerHTML = originalText;
        }
    }

    /**
     * Render comparison table
     */
    function renderComparisonTable(tableData) {
        if (!tableData || tableData.length === 0) {
            elements.chartContainer.innerHTML = '<div class="empty-state"><p>No data available</p></div>';
            return;
        }

        // Get display names for visible variants only
        const visibleVariants = getVisibleTableVariants();
        const variantDisplayNames = visibleVariants.map(v => getDisplayName(v.name));

        if (variantDisplayNames.length === 0) {
            elements.chartContainer.innerHTML = '<div class="empty-state"><p>No variants selected. Please select at least one variant.</p></div>';
            return;
        }

        // Metrics where lower is better
        const lowerBetter = [
            'max_drawdown', 'negative_annualized_sd', 'annualized_sd',
            'rolling_roi_std', 'ulcer_index',
            'weeks_below_x_pct', 'weeks_min_notional_below_y_pct',
            'avg_orders_per_cycle'
        ];

        let html = '<div class="table-wrapper"><table class="comparison-table"><thead><tr><th>Metric</th>';
        variantDisplayNames.forEach(v => {
            html += `<th>${escapeHtml(v)}</th>`;
        });
        html += '</tr></thead><tbody>';

        tableData.forEach(row => {
            // Check if this is a section header row
            if (row.is_section_header) {
                const numCols = variantDisplayNames.length + 1;
                html += `<tr class="section-header-row"><td colspan="${numCols}">${escapeHtml(row.metric)}</td></tr>`;
                return;
            }

            html += `<tr><td>${escapeHtml(row.metric)}</td>`;

            // Find best/worst for this metric (only among visible variants)
            const values = variantDisplayNames.map(v => parseFloat(row[`${v}_raw`]) || 0);
            const isLowerBetter = lowerBetter.includes(row.metric_key);
            const bestVal = isLowerBetter ? Math.min(...values) : Math.max(...values);
            const worstVal = isLowerBetter ? Math.max(...values) : Math.min(...values);

            variantDisplayNames.forEach(v => {
                const rawVal = parseFloat(row[`${v}_raw`]) || 0;
                let className = '';
                if (rawVal === bestVal && values.filter(x => x === bestVal).length === 1) {
                    className = 'best-value';
                } else if (rawVal === worstVal && values.filter(x => x === worstVal).length === 1) {
                    className = 'worst-value';
                }
                html += `<td class="${className}">${escapeHtml(row[v])}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        elements.chartContainer.innerHTML = html;
    }

    /**
     * Render ranking table
     */
    function renderRankingTable(rankingData) {
        if (!rankingData || rankingData.length === 0) {
            elements.chartContainer.innerHTML = '<div class="empty-state"><p>No data available</p></div>';
            return;
        }

        const numVariants = state.variants.length;

        let html = '<div class="table-wrapper"><table class="comparison-table"><thead><tr><th>Metric</th>';
        for (let i = 1; i <= numVariants; i++) {
            html += `<th>#${i}</th>`;
        }
        html += '</tr></thead><tbody>';

        rankingData.forEach(row => {
            html += `<tr><td>${escapeHtml(row.metric)}</td>`;
            for (let i = 1; i <= numVariants; i++) {
                const rankKey = `rank_${i}`;
                html += `<td>${escapeHtml(row[rankKey] || '-')}</td>`;
            }
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        elements.chartContainer.innerHTML = html;
    }

    /**
     * Render Plotly chart
     */
    function renderPlotlyChart(chartData) {
        if (!chartData || !chartData.data) {
            elements.chartContainer.innerHTML = '<div class="empty-state"><p>No chart data available</p></div>';
            return;
        }

        elements.chartContainer.innerHTML = '';

        // Apply theme colors
        const isDark = document.body.classList.contains('dark-theme');
        const layout = chartData.layout || {};

        layout.paper_bgcolor = isDark ? '#1e293b' : '#ffffff';
        layout.plot_bgcolor = isDark ? '#0f172a' : '#f8fafc';
        layout.font = layout.font || {};
        layout.font.color = isDark ? '#f1f5f9' : '#1e293b';

        if (layout.xaxis) {
            layout.xaxis.gridcolor = isDark ? '#334155' : '#e2e8f0';
        }
        if (layout.yaxis) {
            layout.yaxis.gridcolor = isDark ? '#334155' : '#e2e8f0';
        }
        if (layout.polar && layout.polar.radialaxis) {
            layout.polar.radialaxis.gridcolor = isDark ? '#334155' : '#e2e8f0';
        }

        Plotly.newPlot(elements.chartContainer, chartData.data, layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        });
    }

    /**
     * Escape HTML special characters
     */
    function escapeHtml(str) {
        if (typeof str !== 'string') return str;
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Export for external access (theme toggle)
    window.compareApp = {
        state,
        generateChart,
        generateCumulativeChart
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
