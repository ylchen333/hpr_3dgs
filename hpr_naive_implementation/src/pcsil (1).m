function [visInds, silInds] = pcsil(points, view, gamma, sense)

	visInds = hpr(points, view, gamma);
	visPoints = points(visInds, :);
	tpoInds = hpr(visPoints, view, - gamma / sense);
	silInds = visInds;
	silInds(tpoInds) = [];