function OUT = group_analysis_v2_template()
%GROUP_ANALYSIS_V2_TEMPLATE Example modular entry point for posthoc pipeline.
% This file is a non-breaking template intended to guide refactoring of
% group_analysis.m into configurable, registry-driven modules.

    cfg = default_posthoc_config();
    ctx = struct();
    ctx.cfg = cfg;

    registry = register_analyses_template();

    for k = 1:numel(registry)
        mod = registry(k);
        if ~mod.enabled
            continue;
        end

        if ~isempty(mod.compute_fn)
            ctx.metrics.(mod.name) = mod.compute_fn(ctx);
        end

        if ~isempty(mod.stats_fn)
            ctx.stats.(mod.name) = mod.stats_fn(ctx);
        end

        if ~isempty(mod.plot_fn)
            [ctx.figures.(mod.name), ctx.figure_names.(mod.name)] = mod.plot_fn(ctx);
        end
    end

    OUT = ctx;
end

function cfg = default_posthoc_config()
    cfg = struct();
    cfg.cacheDir = './cache';
    cfg.figDir = fullfile('.', '..', '..', '..', 'figures_postproc');
    cfg.groups.exp = {'e21','e22','e25','e26','e29','e30','e31','e32','e38','e39'};
    cfg.groups.ctrl = {'e23','e24','e27','e33','e36','e37'};
end

function registry = register_analyses_template()
% Each analysis module can independently define compute/stats/plot behavior.

    registry = struct('name', {}, 'enabled', {}, 'compute_fn', {}, 'stats_fn', {}, 'plot_fn', {});

    registry(end+1) = struct( ...
        'name', 'rt_stroop', ...
        'enabled', true, ...
        'compute_fn', @compute_rt_stroop_template, ...
        'stats_fn', @stats_rt_stroop_template, ...
        'plot_fn', @plot_rt_stroop_template ...
    );

    registry(end+1) = struct( ...
        'name', 'pd_r2_topos', ...
        'enabled', true, ...
        'compute_fn', @compute_pd_r2_template, ...
        'stats_fn', [], ...
        'plot_fn', @plot_pd_r2_template ...
    );
end

function out = compute_rt_stroop_template(ctx)
% Replace body with calls to your existing helper functions.
    out = struct('note', 'call compute_rt_effects / compute_stroop_effects wrappers here');
    out.groups = fieldnames(ctx.cfg.groups);
end

function out = stats_rt_stroop_template(~)
    out = struct('note', 'call run_stroop_group_anovas or fitrm wrappers here');
end

function [figs, names] = plot_rt_stroop_template(~)
    figs = gobjects(0);
    names = {};
end

function out = compute_pd_r2_template(~)
    out = struct('note', 'call analyze_pd_r2_groups wrappers here');
end

function [figs, names] = plot_pd_r2_template(~)
    figs = gobjects(0);
    names = {};
end
