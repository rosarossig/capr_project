function [lik, latents] = RW3(param, data)
    % framework adapted from Sam Gershman using the mfit package
    % (https://github.com/sjgershm/mfit)

    % N = number of trials, D is number of stimuli 
    [N,D] = size(data.X);

    % make placeholder vector for weights 
    % there is one weight associated with each stimulus
    w = zeros(D,1);       
    % add buffer at the end for updating
    data.X = [data.X; zeros(1,D)];
    % placeholder for likelihood
    lik = 0; 

    % define the parameters 
    alpha = param(1); 
    lambda = param(2);
    gamma = param(3);

    % X is the stimulus vector
    % r is the outcome (allergy or no allergy)
    X = data.X; 
    r = data.r;

    % create placeholders
    rhat_vec = zeros(N,1);
    latents.alpha = NaN*ones(N,2);
    latents.pe = NaN*ones(N,2);

    % run for all the trials 
    for n = 1:N
        % start with isolating the stimulus for this trial
        h = X(n,:);

        % now need to split into single versus compound stimulus
        % start with single 
        if sum(h) == 1
            % create prediction for this stimulus
            % it's literally just the weight for the stimulus
            rhat = h*w;
            
            % calculate prediction error: outcome - prediction
            pe = r(n)-rhat;
            
            % update weight of observed stimulus only
            w = w + alpha*pe*h';
            % save prediction error
            latents.pe(n,1) = pe; 
        else
            % this is for compound stims 
            % start by determining the min and max stims
            stims = find(h == 1);
            stim_1 = stims(1);
            stim_2 = stims(2);
    
            % isolate the weights associated with these stims 
            w1 = w(stim_1);
            w2 = w(stim_2);

            % make weight vector (just in case)
            w_vec = [w1, w2];
  
             % determine the max and min weights/stims    
            max_w = max(w_vec);
            min_w = min(w_vec);


            % 2 is the causal stim
            if max_w == w2
                % start by computing the estimation 
                % ranges from max to full additivity
                % here w2 is the max stim
                rhat = w2+gamma*w1;
                
                % ensure there is no counterfactual updating if both stim
                % have negative values
                if max_w < 0
                    lambda_f = 1;
                else
                    lambda_f = lambda; 
                end

                % calculate the PEs
                pe_max = (r(n)-rhat);
                pe_min = (lambda_f*r(n)-min_w); 

                % here stim 2 is the max stim 
                w(stim_1) = w(stim_1)+alpha*(pe_min);
                w(stim_2) = w(stim_2)+alpha*(pe_max);
                
                % save PEs
                latents.pe(n,1) = pe_max; 
                latents.pe(n,2) = pe_min; 

            else
                rhat = w1+gamma*w2; 

                % ensure no counterfactual learning occurs for negative
                % weights
                if max_w < 0
                    lambda_f = 1;
                else
                    lambda_f = lambda; 
                end
                
                % calculate the PEs
                pe_max = (r(n)-rhat);
                pe_min = (lambda_f*r(n)-min_w);

                % here stim_1 is the max stim
                % update the weights
                w(stim_1) = w(stim_1)+alpha*(pe_max);
                w(stim_2) = w(stim_2)+alpha*(pe_min);

                % save the PEs just in case
                latents.pe(n,1) = pe_max; 
                latents.pe(n,2) = pe_min;
 
            end   
         end

        % save the outputs 
         rhat_vec(n) = rhat; 
         latents.w(n,:) = w;
         latents.rhat(n,1) = rhat;

    
    end
    % using sample variance for response noise instead of another free
    % parameter (as in Rouhani et al., 2021 - https://elifesciences.org/articles/61077)
    var = (nansum((data.response-(rhat_vec)).^2))/(data.N);
    % sum llk - note this function takes in sd, not variance 
    lik = sum(lognormpdf((data.response), rhat_vec, sqrt(var)));
 
     