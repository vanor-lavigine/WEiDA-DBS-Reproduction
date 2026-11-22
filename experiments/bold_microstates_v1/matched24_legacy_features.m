function matched24_legacy_features(root, cache)
% Replay the archived 105 phase features to check the historical focal result.
% The old 105 fit saved assignments and centroids but omitted the feature matrix.
    assert(strcmp(version('-release'),'2020b'));
    folder = fullfile(root,'WEiDA_STN25_HC15_K4_R2020b_20261117');
    author = fullfile(root,'WEiDA-DBS-author-matlab');
    addpath(fullfile(author,'functions'));
    lines = regexp(fileread(fullfile(author,'StateSpace_modelling.m')), '\r?\n','split');
    assert(contains(lines{58},'size(ts_cell)') && startsWith(lines{122},'clear BOLD'));
    load(fullfile(folder,'inputs.mat'),'ts_cell','n_cond');
    rng(0,'twister');
    eval(strjoin(lines(58:121),newline));
    WEiDA_Eig = WEiDA_Eig(1:t_all,:);
    Time_all = Time_all(:,1:t_all);
    native = load(fullfile(folder,'author_k4.mat'),'IDX','Vemp','Time_all');
    assert(isequal(Time_all,native.Time_all) && t_all == 13647);
    [~,labels] = min(pdist2(WEiDA_Eig,native.Vemp,'squaredeuclidean'),[],2);
    assert(isequal(labels,native.IDX));
    save(fullfile(cache,'legacy_native_features.mat'),'WEiDA_Eig','Time_all','-v7');
end
