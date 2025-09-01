function [x_IT_out,x_IV_out,x_IE_out, x_OT_out,x_OV_out,x_OE_out, x_AT,x_AV,x_AE] = makeFrontBehindMiddleSets(x_IT,x_IV,x_IE,x_OT,x_OV,x_OE)
% makeFrontBehindMiddleSets
% -------------------------------------------------------------------------
% Package of your slicing + reshaping routine.
% Splits each [T × N] matrix into front(1:128), middle(129:384), behind(385:512),
% then:
%   - Replaces inputs (IT/IV/IE) with [front_of_target; behind_of_target],
%     passes through extract_reshape.
%   - Replaces targets (OT/OV/OE) with the MIDDLE of the original inputs,
%     passed through reshape_batches.
%   - Builds x_AT/x_AV/x_AE as the MIDDLE of the original targets,
%     passed through reshape_batches (and gather+squeeze for x_AV only,
%     to faithfully match the original snippet).
%
% INPUTS (all required)
%   x_IT, x_IV, x_IE : original input matrices [T × N]
%   x_OT, x_OV, x_OE : original target matrices [T × N]
%
% OUTPUTS
%   x_IT_out, x_IV_out, x_IE_out : inputs after front+behind replacement and extract_reshape
%   x_OT_out, x_OV_out, x_OE_out : targets as middle of original inputs (reshape_batches)
%   x_AT, x_AV, x_AE             : middle of original targets (reshape_batches), with x_AV gathered+squeezed
%
% DEPENDENCIES
%   - extract_reshape.m
%   - reshape_batches.m
%
% NOTES
%   - This function intentionally preserves the exact behavior of your snippet,

    % ---- basic size checks (expect at least 512 rows) ----
    reqRows = 512;
    mats = {x_IT,x_IV,x_IE,x_OT,x_OV,x_OE};
    names = {'x_IT','x_IV','x_IE','x_OT','x_OV','x_OE'};
    for k = 1:numel(mats)
        if size(mats{k},1) < reqRows
            error('%s must have at least %d rows. Got %d.', names{k}, reqRows, size(mats{k},1));
        end
    end

    % ---- define index ranges ----
    frontIdx  = 1:128;
    middleIdx = 129:384;
    behindIdx = 385:512;

    % ---- extract front/behind from TARGETS, middle from INPUTS/TARGETS ----
    x_IT_front   = x_OT(frontIdx,:);
    x_IT_behind  = x_OT(behindIdx,:);
    x_IV_front   = x_OV(frontIdx,:);
    x_IV_behind  = x_OV(behindIdx,:);
    x_IE_front   = x_OE(frontIdx,:);
    x_IE_behind  = x_OE(behindIdx,:);

    x_IT_middle  = x_IT(middleIdx,:);
    x_IV_middle  = x_IV(middleIdx,:);
    x_IE_middle  = x_IE(middleIdx,:);
    x_OT_middle  = x_OT(middleIdx,:);
    x_OV_middle  = x_OV(middleIdx,:);
    x_OE_middle  = x_OE(middleIdx,:);

    % ---- replace inputs with [front; behind] from TARGETS ----
    x_IT_fb = cat(1, x_IT_front, x_IT_behind);
    x_IV_fb = cat(1, x_IV_front, x_IV_behind);
    x_IE_fb = cat(1, x_IE_front, x_IE_behind);

    % ---- reshape inputs via extract_reshape ----
    x_IT_out = extract_reshape(x_IT_fb);
    x_IV_out = extract_reshape(x_IV_fb);
    x_IE_out = extract_reshape(x_IE_fb);

    % ---- targets become MIDDLE of original inputs (reshape_batches) ----
    x_OT_out = reshape_batches(x_IT_middle);
    x_OV_out = reshape_batches(x_IV_middle);
    x_OE_out = reshape_batches(x_IE_middle);

    % ---- auxiliary sets: MIDDLE of original targets (reshape_batches) ----
    x_AT = reshape_batches(x_OT_middle);
    x_AV = reshape_batches(x_OV_middle);
    x_AE = reshape_batches(x_OE_middle);

end
