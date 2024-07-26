Line 1: 't' -- this can be ignored
Line 2: Default allowed constraint violation probability (can be set different)
Line 3: Number of supply locations
Line 4: Number of demand locations (dimension of random vector)
Line 5: Number of demand scenarios
If filetype is gindtrans*, next array is list of probability of each scenario
	Else if filetype is indtrans*, proabilities are assumed to all equal 1/Num scenarios
Next array: Suppply capacities (array of size given in Line 3)
	format is [###, ###, ..., ###]
Next array of arrays: variable cost to ship from each supplier to each demand
	location -- first array is cost to ship from first supplier to all demand
	locations, etc.
Next array of arrays: demand at all locations in all scenarios -- first array is
	demand at the locations under first scenario, etc.
