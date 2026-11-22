function matched24_phase(matchedDir, cache, folds)
% Fit a single condition-blind phase template to both aligned patient pipelines.
% Fold zero uses all patients. Each positive fold omits both visits in both sources.
    assert(strcmp(version('-release'),'2020b') && isfolder(cache));
    sources = {'legacy105','v100_syn9'};
    blocks = cell(2,1);
    people = cell(2,1);
    for i = 1:2
        d = load(fullfile(matchedDir,sources{i},'author_matched_k4.mat'), ...
                 'WEiDA_Eig','Time_all','frames','TR');
        assert(d.TR == 2.1 && isequal(size(d.WEiDA_Eig),[13227,78]));
        scans = double(d.Time_all(1,:))';
        visits = double(d.Time_all(2,:))';
        off = unique(scans(visits == 2));
        on = unique(scans(visits == 3));
        assert(numel(off) == 24 && numel(on) == 24);
        [use,ix] = ismember(scans,[off;on]);
        assert(sum(use) == 10077);
        blocks{i} = d.WEiDA_Eig(use,:);
        people{i} = mod(ix(use)-1,24)+1;
        if i == 1
            axisCheck = d.Time_all;
        else
            assert(isequal(axisCheck,d.Time_all) && isequal(people{1},people{2}));
        end
    end
    X = vertcat(blocks{:});
    person = vertcat(people{:});
    matlab_version = version;
    for fold = folds
        assert(fold >= 0 && fold <= 24 && fold == floor(fold));
        destination = fullfile(cache,sprintf('phase_%02d.mat',fold));
        assert(~isfile(destination),'Output for this fold already exists');
        keep = person ~= fold;
        train = X(keep,:);
        rng(0,'twister');
        [labels,centers,ssd] = kmeans(train,4,'Distance','sqeuclidean', ...
            'Replicates',300,'Display','off','MaxIter',500);
        var_centers = arrayfun(@(v) str2double(sprintf('%.5g',v)), centers);
        counts = accumarray(labels,1,[4,1]);
        for j = 1:4
            assert(max(abs(mean(train(labels==j,:),1)-centers(j,:))) < 1e-10);
        end
        n_train = size(train,1);
        train_people = numel(unique(person(keep)));
        assert(train_people == 24-(fold>0));
        tmp = [destination '.partial.mat'];
        save(tmp,'fold','centers','var_centers','counts','ssd','n_train','train_people','matlab_version','-v7');
        movefile(tmp,destination);
        fprintf('Completed shared phase fit %02d with %d people and %d frames\n',fold,train_people,n_train);
    end
end
