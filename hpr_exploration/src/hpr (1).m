function visiblePtInds = hpr(points, view, param, useLinear)
	if (nargin < 4)
		useLinear = false;
	end
	dim = size(points, 2);
	numPts = size(points, 1);
	% Move the points s.t. C is the origin
	points = points - repmat(view, [numPts 1]);
	% Calculate ||p||
	normp = sqrt(dot(points, points, 2));
	directions = points ./ repmat(normp, [1 dim]);
	% Spherical flipping
	% R = repmat(max(normp) * (10 ^ param), [numPts 1]);
	% P = points + 2 * repmat(R - normp, [1 dim]) .* points ./ repmat(normp, [1 dim]);
	if (useLinear)
		% Use original linear inversion. The below implementation
		% corresponds to using a radius R = 1/2 * (1 / param) * max(norm),
		% thus requires 0 < param < 1.
		P = repmat((1 / param) * max(normp(:)) - normp, [1 dim]) .* directions;
	else
		% Use exponential inversion. The below implementation also requires
		% 0 < param < 1.
		P = repmat(normp .^ (-param), [1 dim]) .* directions;
	end
	%convex hull
	visiblePtInds = unique(convhulln([P; zeros(1, dim)]));
	visiblePtInds(visiblePtInds == numPts + 1) = [];