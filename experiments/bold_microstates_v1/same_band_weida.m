function same_band_weida(dataRoot, cacheDir)
% Private frame-level intermediates for the same-band experiment.
    assert(~isfolder(cacheDir), 'Choose a new private cache directory');
    mkdir(cacheDir);
    author = fullfile(dataRoot, 'WEiDA-DBS-author-matlab');
    addpath(fullfile(author, 'functions'));
    lines = regexp(fileread(fullfile(author, 'StateSpace_modelling.m')), '\r?\n', 'split');
    assert(contains(lines{58}, 'size(ts_cell)') && startsWith(lines{122}, 'clear BOLD'));
    assert(contains(lines{142}, '''Replicates'',300') && contains(lines{142}, '''MaxIter'',500'));
    block = strjoin(lines(58:121), newline);
    anchor = 'Phase_BOLD(seed,:) = angle(hilbert(signal_filt));';
    assert(numel(strfind(block, anchor)) == 1);
    block = strrep(block, anchor, [anchor ' band_wave{s}(seed,:) = signal_filt;']);
    sources = {'105','SyN9'};
    roots = {fullfile(dataRoot,'WEiDA_STN25_HC15_K4_R2020b_20261117'), ...
        fullfile(dataRoot,'V100脑深部刺激WEiDA复现_20261118','代码与辅助')};
    inputs = {'inputs.mat','MATLAB分析输入.mat'};
    originals = {'author_k4.mat','MATLAB四状态分析结果.mat'};
    for iSrc = 1:2
        pack = load(fullfile(roots{iSrc},inputs{iSrc}),'ts_cell');
        native = load(fullfile(roots{iSrc},originals{iSrc}),'IDX','Vemp','Time_all');
        for iBand = 1:2
            if iBand == 1
                bandName = 'primary';
                code = block;
            else
                bandName = 'slow5';
                assert(contains(block,'flp = 0.04;') && contains(block,'fhi = 0.07;'));
                code = strrep(strrep(block,'flp = 0.04;','flp = 0.01;'),'fhi = 0.07;','fhi = 0.027;');
            end
            ts_cell = pack.ts_cell;
            band_wave = cellfun(@(x) zeros(size(x)), ts_cell(:,1),'UniformOutput',false);
            rng(0,'twister');
            eval(code);
            features = WEiDA_Eig(1:t_all,:);
            time_index = Time_all(:,1:t_all);
            assert(isequal(time_index,native.Time_all));
            wave = [band_wave{:}]';
            assert(isequal(size(features),size(wave)) && all(isfinite(features(:))) && all(isfinite(wave(:))));
            if iBand == 1
                labels = native.IDX(:);
                centers = native.Vemp;
                var_centers = csvread(fullfile(roots{iSrc},'WEiDA4_atlasAAL78_DBS','Vemp.csv'))';
            else
                [labels,centers] = kmeans(features,4,'Distance','sqeuclidean','Replicates',300,'Display','off','MaxIter',500);
                % Match the five-significant-digit export read by the author's local VAR script.
                var_centers = arrayfun(@(v) str2double(sprintf('%.5g',v)),centers);
            end
            matlab_version = version;
            save(fullfile(cacheDir,[sources{iSrc} '_' bandName '.mat']), ...
                 'wave','features','time_index','labels','centers','var_centers','matlab_version','-v7');
            fprintf('Completed %s %s: %d frames, %d scans\n',sources{iSrc},bandName,t_all,size(ts_cell,1));
        end
    end
end
