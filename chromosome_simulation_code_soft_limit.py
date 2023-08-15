import numpy as np
import itertools
import random
import math
from numba import jit
from numba.typed import List


#@jit
@jit(nopython=True)
def rand_choice_nb(choiceArr,probArr):
	cumulative_distribution = np.cumsum(probArr)
	cumulative_distribution /= cumulative_distribution[-1]
	sample = np.random.rand()
	index = np.searchsorted(cumulative_distribution,sample,side ="right")
	return index



def flatten(arr):
	return np.array([item for sublist in arr for item in sublist])
	
		
 ### calculates the mean number of chromosomes observed among the sample points in carr
 ### note this is based on # occurences, NOT holding time, which it probably should be based on instead. 


def get_sim_mean_number(carr):

	return np.mean(np.apply_along_axis(np.sum, -1, carr!=0))


### gets the minimum chromsome size for a given list of chromosome sizes at a single sampe point
def get_min_chr_size(sizeArray):
	arr = sizeArray[sizeArray!=0]
	return arr.min()

### gets the minimum chromsome size for a given list of chromosome sizes at a single sample point
#@jit
@jit(nopython=True)
def get_max_chr_size(sizeArray):
	arr = sizeArray[sizeArray!=0]
	return arr.max()


### get the number of chromosomes given an array of sizes (which may/will contain 0s)
#@jit
@jit(nopython=True)
def get_num_chrs(sizeArray):
	arr =sizeArray!=0
	return sum(arr)
	#return np.apply_along_axis(np.sum, -1, sizeArray!=0)[0]

### get the chromosome sizes (removing 0 entries) and sort them smallest to biggest
### returns a numpy array the length of the non-zero chromsome sizes
#@jit
@jit(nopython=True)
def get_ordered_chromosome_sizes(sizeArray):
	sizes = sizeArray[sizeArray!=0]
	sizes = np.sort(sizes)
	return(sizes)

### this function allows us to do a linear interpolation for chromosome number as a function of beta (and alpha)
### we can do a first minimal test to see how num of chromosomes varies as a function of beta
### using interpolation, we can determine which values of beta to simulate extensively in a second round
### in order to get an uniform distribution of samples across the number of chromsomes observed from k = 2, 3, ... k_max, where 
### kmax will be determined by the limit in the initial test (to avoid extrapolation) 
def linear_interpolate(xVals,yVals,yTarget):
	m = -1
	intervals = list(zip( xVals[0:-1], xVals[1:],yVals[0:-1], yVals[1:]))
	for (x1,x2,y1,y2) in intervals:
		if y1<=yTarget:
			if yTarget<y2:
				m = (y2 - y1)/(x2 - x1)
				break
	if m == -1:
		print(f"could not interpolate k = {yTarget} chromsomes")
		return(-1)
	
	z = (yTarget - y1)/m + x1
	return(z)


def get_square_distance(observed,expected):
    xs = [(a-b)**2 for a,b in list(zip(observed,expected))]
    return sum(xs)



### calculate and return the stationary distribution of chromsome numbers for a given beta and alpha 
### under a simple model where breaks occur at rate beta per chromsome
def stationary_distribution(beta,alpha):
	maxNumber = 100;
	beta = beta
	alpha = 2*alpha # this makes the total break rate in this model correspond to simulations.
	x1 = 1/ ( alpha/(2*beta)*(math.exp(2*beta/alpha)-1))
	x2Onward = [ x1* alpha/(2*beta) * (2* beta/ alpha)**k / math.factorial(k) for k in range (2,maxNumber)] 
	res = [x1] + x2Onward
	return(res)


### correctly calculate the probability of observing n chromsomes for a simulation with a given beta and alpha
### the distribution is calculated from the holding-times for observing k chromsomes, NOT from how OFTEN we observe k chromsomes 
def parallel_get_prob_observing_n_chr(arr):
	a,b = arr
	return(	get_prob_observing_n_chrs(a,b) )


#@jit(nopython=True)
def get_prob_observing_n_chrs(chrsArr,timesArr):
	#nChrs = np.apply_along_axis(get_num_chrs,-1,chrsArr)
	nChrs = [get_num_chrs(x) for x in chrsArr];
	xVals = np.unique(nChrs)
	totalTimes =[timesArr[nChrs == x] for x in xVals]
	yVals = np.array([np.sum(x) for x in totalTimes])
	yVals = yVals/np.sum(yVals)

	return(xVals,yVals)


@jit(nopython=True)
def get_mean_number_chrs(nChrList,nProbList):
	# nChrList is a list-like of the number of chromosomes
	# nProbList is a list-like of the corresponding probabilities
	avgNumb = np.sum([a*b for a,b in list(zip(nChrList,nProbList))])
	return(avgNumb)



@jit(nopython=True)
def get_flat_nChr_array(arrArr):
	nChr = [[get_num_chrs(x2) for x2 in x1] for x1 in arrArr]
	# for x1 in arrArr:
	# 	nChr.append([get_num_chrs(x2) for x2 in x1])
	nChr = [item for sublist in nChr for item in sublist]
	return(np.array(nChr))


### given a set of observations of chromosome sizes, extract the entries where number of chromsomes matches the target, and 
### order the chromsomes by their sizes from smallest to largest
### finally, take the transpose. It returns a list of lists: [[smallestChromosomeSizes],[secondSmallestChromosomeSizes],etc.]
### this is the format needed to obtain mean chromosomes sizes and to generate a box plot
#@jit(nopython=True)
def get_size_distribution_given_n_chrs(chrSizeObservationsArray,targetNum):
	"""
	chrSizeObservations is a flat array, each entry is the array of observed chromosome sizes
	targetNum is the number of chromosomes we want our samples to have
	"""
	#nChrs = np.apply_along_axis(get_num_chrs,-1,chrSizeObservationsArray)
	nChrs = np.array([get_num_chrs(x) for x in chrSizeObservationsArray])
	samples = chrSizeObservationsArray[nChrs==targetNum]
	samples = np.array([get_ordered_chromosome_sizes(x) for x in samples])
	#samples = np.apply_along_axis(get_ordered_chromosome_sizes,-1,samples)
	#samples = np.array(samples)
	return samples.transpose()








#### m0 : Equal Break, Equal Fuse ####


def parallel_simulate_eb_ef(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_eb_ef(numSamplePoints,jumpSize,beta,alpha) )



### In this simulation model, each chromsome breaks with rate beta and any two (unique) chromosomes fuse with rate alpha
@jit(nopython=True)
def simulate_eb_ef(numSamplePoints,jumpSize,beta,alpha):

	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)
	beta = float(beta)
	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps

	for record_idx in range(numSamplePoints):

		for itt1 in range(jumpSize):

			currentChrLengths = newChrLengths[:]

			cLen=len(currentChrLengths)

			if cLen == 1:
				bPoint = np.random.rand()
				newChrLengths = [bPoint,1-bPoint]
				bRates=[beta]
				fRates=np.array([0.0])
				eventRates = np.array(bRates)*1.0

			else:

				#break rates

				bRates = np.ones(cLen)*beta

				cIdxs =range(cLen)
				choices = [[(a,b) for b in cIdxs] for a in cIdxs]
				choices = [item for sublist in choices for item in sublist]

				fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
				#fRates =np.array( [0 if i==j else currentChrLengths[i]/currentChrLengths[j] for (i,j) in choices])*alpha
				#fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

				bTotal=np.sum(bRates)
				fTotal = np.sum(fRates)
				eventRates = np.array([bTotal,fTotal])
				eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

				event = rand_choice_nb(np.array([0,1]),eventWeights)

				if event == 0:
					weights = bRates/sum(bRates)
					choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

					oldChrLengths = currentChrLengths[:]
					old = oldChrLengths.pop(choiceIdx)
					bPoint = np.random.rand()
					bPoint = bPoint * old
					newChrLengths = oldChrLengths + [bPoint,old-bPoint]

				if event == 1:

					weights = fRates/sum(fRates)
					pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
					leftIdx,rightIdx = choices[pairIdx]
					left = currentChrLengths[leftIdx]
					right = currentChrLengths[rightIdx]

					newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
					newChrLengths = newChrLengths + [(left+right)]


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.random.exponential(1/totalRate)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)



### parallel wrapper for model with per-chromsome break rate beta and pair-wise fusion rate alpha AND minimum and max relative chr sizes

def parallel_simulate_eb_ef_softLimit(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_eb_ef_softLimit(numSamplePoints,jumpSize,beta,alpha) )



### simulates a model with per-hromsome break rate beta and pair-wise fusion rate alpha
### there is a hard-coded limit to sizes. 
### No chromosome can be generated that is smaller than relativeMin*averageLength
### No chromsome can be generated that is larger than relativeMax*averageLength
### these values come from a paper showing their relative universality
@jit(nopython=True)
def simulate_eb_ef_softLimit(numSamplePoints,jumpSize,beta,alpha):


	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)

	beta = float(beta)   
	alpha = float(alpha)

	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps


	for record_idx in range(numSamplePoints):



		for itt1 in range(jumpSize):


			currentChrLengths = newChrLengths[:]


			cLen=len(currentChrLengths)

			number_attempts= 0
			while True:
				number_attempts+=1

				if cLen == 1:
					bPoint = np.random.rand()
					newChrLengths = [bPoint,1-bPoint]
					bRates=[beta]
					fRates=np.array([0.0])
					eventRates = np.array(bRates)*1.0

				else:

					#break rates

					bRates = np.ones(cLen)*beta
					


					cIdxs =range(cLen)
					choices = [[(a,b) for b in cIdxs] for a in cIdxs]
					choices = [item for sublist in choices for item in sublist]
					fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
					#fRates =np.array( [0 if i==j else currentChrLengths[i]/currentChrLengths[j] for (i,j) in choices])*alpha
					#fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

					bTotal=np.sum(bRates)
					fTotal = np.sum(fRates)
					eventRates = np.array([bTotal,fTotal])*1.0
					eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

					event = rand_choice_nb(np.array([0,1]),eventWeights)

					if event == 0:
						weights = bRates/sum(bRates)
						choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

						oldChrLengths = currentChrLengths[:]
						old = oldChrLengths.pop(choiceIdx)
						bPoint = np.random.rand()
						bPoint = bPoint * old
						newChrLengths = oldChrLengths + [bPoint,old-bPoint]

					if event == 1:

						weights = fRates/sum(fRates)
						pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
						leftIdx,rightIdx = choices[pairIdx]
						left = currentChrLengths[leftIdx]
						right = currentChrLengths[rightIdx]

						newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
						newChrLengths = newChrLengths + [(left+right)]
				

				kCurrent = len(currentChrLengths)
				optCurrent = np.prod(np.array([1-math.exp(-1/kCurrent) for x in range(kCurrent)]))
				obsCurrent = np.prod(np.array([1-math.exp(-x) for x in currentChrLengths]))
				relCurrent = obsCurrent/optCurrent

				kNew = len(newChrLengths)
				optNew = np.prod(np.array([1-math.exp(-1/kNew) for x in range(kNew)]))
				obsNew = np.prod(np.array([1 - math.exp(-x) for x in newChrLengths]))
				relNew = obsNew/optNew

				if relNew >= relCurrent:
					break
				else:
					ranDraw = np.random.rand()
					if ranDraw <= relNew/relCurrent:
						break



	
		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.array([np.random.exponential(1/totalRate) for x in range(number_attempts)])
		expTime = np.sum(expTime)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    







#### m1: Propotrional Break, Equal Fuse #### 

def parallel_simulate_pb_ef(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_pb_ef(numSamplePoints,jumpSize,beta,alpha) )


@jit(nopython=True)
def simulate_pb_ef(numSamplePoints,jumpSize,beta,alpha):

	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)
	beta = float(beta)   
	alpha = float(alpha)
	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps


	for record_idx in range(numSamplePoints):



		for itt1 in range(jumpSize):


			currentChrLengths = newChrLengths[:]


			cLen=len(currentChrLengths)

			if cLen == 1:
				bPoint = np.random.rand()
				newChrLengths = [bPoint,1-bPoint]
				bRates=[beta]
				fRates=np.array([0.0])
				eventRates = np.array(bRates)*1.0

			else:

				#break rates

				#bRates = np.ones(cLen)*beta
				bRates = np.array(currentChrLengths)*beta


				cIdxs =range(cLen)
				choices = [[(a,b) for b in cIdxs] for a in cIdxs]
				choices = [item for sublist in choices for item in sublist]
				fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
				#fRates =np.array( [0 if i==j else currentChrLengths[i]/currentChrLengths[j] for (i,j) in choices])*alpha
				#fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

				bTotal=np.sum(bRates)
				fTotal = np.sum(fRates)
				eventRates = np.array([bTotal,fTotal])*1.0
				eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

				event = rand_choice_nb(np.array([0,1]),eventWeights)

				if event == 0:
					weights = bRates/sum(bRates)
					choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

					oldChrLengths = currentChrLengths[:]
					old = oldChrLengths.pop(choiceIdx)
					bPoint = np.random.rand()
					bPoint = bPoint * old
					newChrLengths = oldChrLengths + [bPoint,old-bPoint]

				if event == 1:

					weights = fRates/sum(fRates)
					pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
					leftIdx,rightIdx = choices[pairIdx]
					left = currentChrLengths[leftIdx]
					right = currentChrLengths[rightIdx]

					newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
					newChrLengths = newChrLengths + [(left+right)]


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.random.exponential(1/totalRate)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    
	

def parallel_simulate_pb_ef_softLimit(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_pb_ef_softLimit(numSamplePoints,jumpSize,beta,alpha) )



### simulates a model with proportional break rate beta*chr Length and pair-wise fusion rate alpha
### there is a hard-coded limit to sizes. 
### No chromosome can be generated that is smaller than relativeMin*averageLength
### No chromsome can be generated that is larger than relativeMax*averageLength
### these values come from a paper showing their relative universality


@jit(nopython=True)
def simulate_pb_ef_softLimit(numSamplePoints,jumpSize,beta,alpha):


	beta = float(beta)   
	alpha = float(alpha)
	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)

	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps


	for record_idx in range(numSamplePoints):



		for itt1 in range(jumpSize):


			currentChrLengths = newChrLengths[:]


			cLen=len(currentChrLengths)
			number_attempts = 0
			while True:
				number_attempts += 1
				if cLen == 1:
					bPoint = np.random.rand()
					newChrLengths = [bPoint,1-bPoint]
					bRates=[beta]
					fRates=np.array([0.0])
					eventRates = np.array(bRates)*1.0

				else:

					#break rates

					#bRates = np.ones(cLen)*beta
					bRates = np.array(currentChrLengths)*beta


					cIdxs =range(cLen)
					choices = [[(a,b) for b in cIdxs] for a in cIdxs]
					choices = [item for sublist in choices for item in sublist]

					fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
					#fRates =np.array( [0 if i==j else currentChrLengths[i]/currentChrLengths[j] for (i,j) in choices])*alpha
					#fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

					bTotal=np.sum(bRates)
					fTotal = np.sum(fRates)
					eventRates = np.array([bTotal,fTotal])*1.0
					eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

					event = rand_choice_nb(np.array([0,1]),eventWeights)

					if event == 0:
						weights = bRates/sum(bRates)
						choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

						oldChrLengths = currentChrLengths[:]
						old = oldChrLengths.pop(choiceIdx)
						bPoint = np.random.rand()
						bPoint = bPoint * old
						newChrLengths = oldChrLengths + [bPoint,old-bPoint]

					if event == 1:

						weights = fRates/sum(fRates)
						pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
						leftIdx,rightIdx = choices[pairIdx]
						left = currentChrLengths[leftIdx]
						right = currentChrLengths[rightIdx]

						newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
						newChrLengths = newChrLengths + [(left+right)]



				kCurrent = len(currentChrLengths)
				optCurrent = np.prod(np.array([1-math.exp(-1/kCurrent) for x in range(kCurrent)]))
				obsCurrent = np.prod(np.array([1-math.exp(-x) for x in currentChrLengths]))
				relCurrent = obsCurrent/optCurrent

				kNew = len(newChrLengths)
				optNew = np.prod(np.array([1-math.exp(-1/kNew) for x in range(kNew)]))
				obsNew = np.prod(np.array([1 - math.exp(-x) for x in newChrLengths]))
				relNew = obsNew/optNew

				if relNew >= relCurrent:
					break
				else:
					ranDraw = np.random.rand()
					if ranDraw <= relNew/relCurrent:
						break


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.array([np.random.exponential(1/totalRate) for x in range(number_attempts)])
		expTime = np.sum(expTime)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    









#### m2: Equal Break, Proportional Fuse #### 

def parallel_simulate_eb_pf(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_eb_pf(numSamplePoints,jumpSize,beta,alpha) )


	
@jit(nopython=True)
def simulate_eb_pf(numSamplePoints,jumpSize,beta,alpha):

	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)
	beta = float(beta)
	alpha = float(alpha)
	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps


	for record_idx in range(numSamplePoints):



		for itt1 in range(jumpSize):


			currentChrLengths = newChrLengths[:]


			cLen=len(currentChrLengths)

			if cLen == 1:
				#bPoint = np.random.uniform(low=0,high=1)
				bPoint = np.random.rand()

				newChrLengths = [bPoint,1-bPoint]
				bRates=[beta]
				fRates=np.array([0.0])
				eventRates = np.array(bRates)*1.0
			else:

				#break rates

				bRates = np.ones(cLen)*beta*1.0


				cIdxs =list(range(cLen))
				#choices = list(itertools.product(cIdxs,cIdxs))
				choices = [[(a,b) for b in cIdxs] for a in cIdxs]
				choices = [item for sublist in choices for item in sublist]

				#fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
				fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

				bTotal=np.sum(bRates)
				fTotal = np.sum(fRates)
				eventRates = np.array([bTotal,fTotal])*1.0
				eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

				#event = np.random.choice([0,1],p=eventWeights)
				event = rand_choice_nb(np.array([0,1]),eventWeights)

				if event == 0:
					weights = bRates/sum(bRates)

					#choiceIdx = np.random.choice(range(cLen),1,p=weights)[0]
					choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)
					
					oldChrLengths = currentChrLengths[:]
					old = oldChrLengths.pop(choiceIdx)
					bPoint = np.random.rand()
					bPoint = bPoint * old
					#bPoint = np.random.uniform(low = 0, high = old)
					newChrLengths = oldChrLengths + [bPoint,old-bPoint]

				if event == 1:

					weights = fRates/sum(fRates)
					pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)


					#pairIdx = np.random.choice(range(len(choices)), 1, p = weights)[0]
					leftIdx,rightIdx = choices[pairIdx]
					left = currentChrLengths[leftIdx]
					right = currentChrLengths[rightIdx]

					newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
					newChrLengths = newChrLengths + [(left+right)]


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.random.exponential(1/totalRate)
		
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    



def parallel_simulate_eb_pf_softLimit(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_eb_pf_softLimit(numSamplePoints,jumpSize,beta,alpha) )



### simulates a model with proportional break rate beta*chr Length and pair-wise fusion rate alpha
### there is a hard-coded limit to sizes. 
### No chromosome can be generated that is smaller than relativeMin*averageLength
### No chromsome can be generated that is larger than relativeMax*averageLength
### these values come from a paper showing their relative universality


@jit(nopython=True)
def simulate_eb_pf_softLimit(numSamplePoints,jumpSize,beta,alpha):


	beta = float(beta)
	alpha = float(alpha)
	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)

	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps

	for record_idx in range(numSamplePoints):

		for itt1 in range(jumpSize):

			currentChrLengths = newChrLengths[:]

			cLen=len(currentChrLengths)
			number_attempts = 0
			while True:
				number_attempts+=1


				if cLen == 1:
					#bPoint = np.random.uniform(low=0,high=1)
					bPoint = np.random.rand()
					newChrLengths = [bPoint,1-bPoint]
					bRates=np.array([beta])
					fRates=np.array([0.0])
					eventRates = bRates*1.0

				else:

					#break rates

					bRates = np.ones(cLen)*beta
					#bRates = np.array(currentChrLengths)*beta


					cIdxs =list(range(cLen))
					choices = [[(a,b) for b in cIdxs] for a in cIdxs]
					choices = [item for sublist in choices for item in sublist]
					#fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
					
					fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

					bTotal=np.sum(bRates)
					fTotal = np.sum(fRates)
					eventRates = np.array([bTotal,fTotal])
					eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

					event = rand_choice_nb(np.array([0,1]),eventWeights)

					if event == 0:
						weights = bRates/sum(bRates)
						choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

						oldChrLengths = currentChrLengths[:]
						old = oldChrLengths.pop(choiceIdx)
						bPoint = np.random.rand()
						bPoint = bPoint * old
						#bPoint = np.random.uniform(low = 0, high = old)
						newChrLengths = oldChrLengths + [bPoint,old-bPoint]

					if event == 1:

						weights = fRates/sum(fRates)
						pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
						leftIdx,rightIdx = choices[pairIdx]
						left = currentChrLengths[leftIdx]
						right = currentChrLengths[rightIdx]

						newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
						newChrLengths = newChrLengths + [(left+right)]


				kCurrent = len(currentChrLengths)
				optCurrent = np.prod(np.array([1-math.exp(-1/kCurrent) for x in range(kCurrent)]))
				obsCurrent = np.prod(np.array([1-math.exp(-x) for x in currentChrLengths]))
				relCurrent = obsCurrent/optCurrent

				kNew = len(newChrLengths)
				optNew = np.prod(np.array([1-math.exp(-1/kNew) for x in range(kNew)]))
				obsNew = np.prod(np.array([1 - math.exp(-x) for x in newChrLengths]))
				relNew = obsNew/optNew

				if relNew >= relCurrent:
					break
				else:
					ranDraw = np.random.rand()
					if ranDraw <= relNew/relCurrent:
						break



		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.array([np.random.exponential(1/totalRate) for x in range(number_attempts)])
		expTime = np.sum(expTime)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    










##### m3: proportional break, proportional fuse

def parallel_simulate_pb_pf(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_pb_pf(numSamplePoints,jumpSize,beta,alpha) )


	
@jit(nopython=True)
def simulate_pb_pf(numSamplePoints,jumpSize,beta,alpha):

	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)
	beta = float(beta)
	alpha = float(alpha)
	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps


	for record_idx in range(numSamplePoints):



		for itt1 in range(jumpSize):


			currentChrLengths = newChrLengths[:]


			cLen=len(currentChrLengths)

			if cLen == 1:
				#bPoint = np.random.uniform(low=0,high=1)
				bPoint = np.random.rand()

				newChrLengths = [bPoint,1-bPoint]
				bRates=[beta]
				fRates=np.array([0.0])
				eventRates = np.array(bRates)*1.0
			else:

				#break rates
				bRates = np.array(currentChrLengths)*beta


				cIdxs =list(range(cLen))
				#choices = list(itertools.product(cIdxs,cIdxs))
				choices = [[(a,b) for b in cIdxs] for a in cIdxs]
				choices = [item for sublist in choices for item in sublist]

				#fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
				fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

				bTotal=np.sum(bRates)
				fTotal = np.sum(fRates)
				eventRates = np.array([bTotal,fTotal])*1.0
				eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

				#event = np.random.choice([0,1],p=eventWeights)
				event = rand_choice_nb(np.array([0,1]),eventWeights)

				if event == 0:
					weights = bRates/sum(bRates)

					#choiceIdx = np.random.choice(range(cLen),1,p=weights)[0]
					choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)
					
					oldChrLengths = currentChrLengths[:]
					old = oldChrLengths.pop(choiceIdx)
					bPoint = np.random.rand()
					bPoint = bPoint * old
					#bPoint = np.random.uniform(low = 0, high = old)
					newChrLengths = oldChrLengths + [bPoint,old-bPoint]

				if event == 1:

					weights = fRates/sum(fRates)
					pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)


					#pairIdx = np.random.choice(range(len(choices)), 1, p = weights)[0]
					leftIdx,rightIdx = choices[pairIdx]
					left = currentChrLengths[leftIdx]
					right = currentChrLengths[rightIdx]

					newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
					newChrLengths = newChrLengths + [(left+right)]


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.random.exponential(1/totalRate)
		
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    



def parallel_simulate_pb_pf_softLimit(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_pb_pf_softLimit(numSamplePoints,jumpSize,beta,alpha) )




@jit(nopython=True)
def simulate_pb_pf_softLimit(numSamplePoints,jumpSize,beta,alpha):


	beta = float(beta)
	alpha = float(alpha)
	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)

	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps

	for record_idx in range(numSamplePoints):

		for itt1 in range(jumpSize):

			currentChrLengths = newChrLengths[:]

			cLen=len(currentChrLengths)
			number_attempts = 0
			while True:
				number_attempts+=1


				if cLen == 1:
					#bPoint = np.random.uniform(low=0,high=1)
					bPoint = np.random.rand()
					newChrLengths = [bPoint,1-bPoint]
					bRates=np.array([beta])
					fRates=np.array([0.0])
					eventRates = bRates*1.0

				else:

					#break rates

					bRates = np.array(currentChrLengths)*beta
					


					cIdxs =list(range(cLen))
					choices = [[(a,b) for b in cIdxs] for a in cIdxs]
					choices = [item for sublist in choices for item in sublist]
					#fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
					
					fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

					bTotal=np.sum(bRates)
					fTotal = np.sum(fRates)
					eventRates = np.array([bTotal,fTotal])
					eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

					event = rand_choice_nb(np.array([0,1]),eventWeights)

					if event == 0:
						weights = bRates/sum(bRates)
						choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

						oldChrLengths = currentChrLengths[:]
						old = oldChrLengths.pop(choiceIdx)
						bPoint = np.random.rand()
						bPoint = bPoint * old
						#bPoint = np.random.uniform(low = 0, high = old)
						newChrLengths = oldChrLengths + [bPoint,old-bPoint]

					if event == 1:

						weights = fRates/sum(fRates)
						pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
						leftIdx,rightIdx = choices[pairIdx]
						left = currentChrLengths[leftIdx]
						right = currentChrLengths[rightIdx]

						newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
						newChrLengths = newChrLengths + [(left+right)]
				

				kCurrent = len(currentChrLengths)
				optCurrent = np.prod(np.array([1-math.exp(-1/kCurrent) for x in range(kCurrent)]))
				obsCurrent = np.prod(np.array([1-math.exp(-x) for x in currentChrLengths]))
				relCurrent = obsCurrent/optCurrent

				kNew = len(newChrLengths)
				optNew = np.prod(np.array([1-math.exp(-1/kNew) for x in range(kNew)]))
				obsNew = np.prod(np.array([1 - math.exp(-x) for x in newChrLengths]))
				relNew = obsNew/optNew

				if relNew >= relCurrent:
					break
				else:
					ranDraw = np.random.rand()
					if ranDraw <= relNew/relCurrent:
						break


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.array([np.random.exponential(1/totalRate) for x in range(number_attempts)])
		expTime = np.sum(expTime)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    










##### m4: Equal Break, Short Fuse Big Stick

def parallel_simulate_eb_sfbs(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_eb_sfbs(numSamplePoints,jumpSize,beta,alpha) )


@jit(nopython=True)	
def simulate_eb_sfbs(numSamplePoints,jumpSize,beta,alpha):


	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)
	beta = float(beta)
	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps

	for record_idx in range(numSamplePoints):

		for itt1 in range(jumpSize):

			currentChrLengths = newChrLengths[:]

			cLen=len(currentChrLengths)

			if cLen == 1:
				bPoint = np.random.rand()
				newChrLengths = [bPoint,1-bPoint]
				bRates=[beta]
				fRates=np.array([0.0])
				eventRates = np.array(bRates)*1.0

			else:

				#break rates

				bRates = np.ones(cLen)*beta

				cIdxs =range(cLen)
				choices = [[(a,b) for b in cIdxs] for a in cIdxs]
				choices = [item for sublist in choices for item in sublist]

				fRates = np.array( [0 if i==j else 1/((1 - abs(currentChrLengths[i]-currentChrLengths[j]))**cLen) for (i,j) in choices])*alpha

				#fRates = np.array( [0 if i==j else 1 for (i,j) in choices])*alpha
				#fRates =np.array( [0 if i==j else currentChrLengths[i]/currentChrLengths[j] for (i,j) in choices])*alpha
				#fRates = np.array( [0 if i==j else 1/currentChrLengths[i]*1/currentChrLengths[j] for (i,j) in choices])*alpha

				bTotal=np.sum(bRates)
				fTotal = np.sum(fRates)
				eventRates = np.array([bTotal,fTotal])
				eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

				event = rand_choice_nb(np.array([0,1]),eventWeights)

				if event == 0:
					weights = bRates/sum(bRates)
					choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

					oldChrLengths = currentChrLengths[:]
					old = oldChrLengths.pop(choiceIdx)
					bPoint = np.random.rand()
					bPoint = bPoint * old
					newChrLengths = oldChrLengths + [bPoint,old-bPoint]

				if event == 1:

					weights = fRates/sum(fRates)
					pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
					leftIdx,rightIdx = choices[pairIdx]
					left = currentChrLengths[leftIdx]
					right = currentChrLengths[rightIdx]

					newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
					newChrLengths = newChrLengths + [(left+right)]


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.random.exponential(1/totalRate)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)



def parallel_simulate_eb_sfbs_softLimit(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_eb_sfbs_softLimit(numSamplePoints,jumpSize,beta,alpha) )

@jit(nopython=True)
def simulate_eb_sfbs_softLimit(numSamplePoints,jumpSize,beta,alpha):


	beta = float(beta)
	alpha = float(alpha)
	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)

	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps

	for record_idx in range(numSamplePoints):

		for itt1 in range(jumpSize):

			currentChrLengths = newChrLengths[:]

			cLen=len(currentChrLengths)
			number_attempts = 0
			while True:
				number_attempts+=1


				if cLen == 1:
					#bPoint = np.random.uniform(low=0,high=1)
					bPoint = np.random.rand()
					newChrLengths = [bPoint,1-bPoint]
					bRates=np.array([beta])
					fRates=np.array([0.0])
					eventRates = bRates*1.0

				else:

					#break rates

					bRates = np.ones(cLen)*beta
					


					cIdxs =list(range(cLen))
					choices = [[(a,b) for b in cIdxs] for a in cIdxs]
					choices = [item for sublist in choices for item in sublist]

					fRates = np.array( [0 if i==j else 1/((1 - abs(currentChrLengths[i]-currentChrLengths[j]))**cLen) for (i,j) in choices])*alpha

					bTotal=np.sum(bRates)
					fTotal = np.sum(fRates)
					eventRates = np.array([bTotal,fTotal])
					eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

					event = rand_choice_nb(np.array([0,1]),eventWeights)

					if event == 0:
						weights = bRates/sum(bRates)
						choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

						oldChrLengths = currentChrLengths[:]
						old = oldChrLengths.pop(choiceIdx)
						bPoint = np.random.rand()
						bPoint = bPoint * old
						#bPoint = np.random.uniform(low = 0, high = old)
						newChrLengths = oldChrLengths + [bPoint,old-bPoint]

					if event == 1:

						weights = fRates/sum(fRates)
						pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
						leftIdx,rightIdx = choices[pairIdx]
						left = currentChrLengths[leftIdx]
						right = currentChrLengths[rightIdx]

						newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
						newChrLengths = newChrLengths + [(left+right)]



				kCurrent = len(currentChrLengths)
				optCurrent = np.prod(np.array([1-math.exp(-1/kCurrent) for x in range(kCurrent)]))
				obsCurrent = np.prod(np.array([1-math.exp(-x) for x in currentChrLengths]))
				relCurrent = obsCurrent/optCurrent

				kNew = len(newChrLengths)
				optNew = np.prod(np.array([1-math.exp(-1/kNew) for x in range(kNew)]))
				obsNew = np.prod(np.array([1 - math.exp(-x) for x in newChrLengths]))
				relNew = obsNew/optNew

				if relNew >= relCurrent:
					break
				else:
					ranDraw = np.random.rand()
					if ranDraw <= relNew/relCurrent:
						break





		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.array([np.random.exponential(1/totalRate) for x in range(number_attempts)])
		expTime = np.sum(expTime)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    







##### m5: Proportional Break, Short Fuse Big Stick

def parallel_simulate_pb_sfbs(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_pb_sfbs(numSamplePoints,jumpSize,beta,alpha) )


@jit(nopython=True)	
def simulate_pb_sfbs(numSamplePoints,jumpSize,beta,alpha):


	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)
	beta = float(beta)
	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps

	for record_idx in range(numSamplePoints):

		for itt1 in range(jumpSize):

			currentChrLengths = newChrLengths[:]

			cLen=len(currentChrLengths)

			if cLen == 1:
				bPoint = np.random.rand()
				newChrLengths = [bPoint,1-bPoint]
				bRates=[beta]
				fRates=np.array([0.0])
				eventRates = np.array(bRates)*1.0

			else:

				#break rates

				# bRates = np.ones(cLen)*beta
				bRates = np.array(currentChrLengths)*beta

				cIdxs =range(cLen)
				choices = [[(a,b) for b in cIdxs] for a in cIdxs]
				choices = [item for sublist in choices for item in sublist]

				fRates = np.array( [0 if i==j else 1/((1 - abs(currentChrLengths[i]-currentChrLengths[j]))**cLen) for (i,j) in choices])*alpha


				bTotal=np.sum(bRates)
				fTotal = np.sum(fRates)
				eventRates = np.array([bTotal,fTotal])
				eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

				event = rand_choice_nb(np.array([0,1]),eventWeights)

				if event == 0:
					weights = bRates/sum(bRates)
					choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

					oldChrLengths = currentChrLengths[:]
					old = oldChrLengths.pop(choiceIdx)
					bPoint = np.random.rand()
					bPoint = bPoint * old
					newChrLengths = oldChrLengths + [bPoint,old-bPoint]

				if event == 1:

					weights = fRates/sum(fRates)
					pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
					leftIdx,rightIdx = choices[pairIdx]
					left = currentChrLengths[leftIdx]
					right = currentChrLengths[rightIdx]

					newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
					newChrLengths = newChrLengths + [(left+right)]


		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.random.exponential(1/totalRate)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)



def parallel_simulate_pb_sfbs_softLimit(varList):
	numSamplePoints,jumpSize,beta,alpha = varList
	return ( simulate_pb_sfbs_softLimit(numSamplePoints,jumpSize,beta,alpha) )

@jit(nopython=True)
def simulate_pb_sfbs_softLimit(numSamplePoints,jumpSize,beta,alpha):


	beta = float(beta)
	alpha = float(alpha)
	resChrs = np.zeros((numSamplePoints,200))
	resTime = np.zeros(numSamplePoints)

	newChrLengths = [1.0]  #used to initilize the 'current' state at the beginning of each loop
	# rather than the end so we can record things correctly between jumps

	for record_idx in range(numSamplePoints):

		for itt1 in range(jumpSize):

			currentChrLengths = newChrLengths[:]

			cLen=len(currentChrLengths)
			number_attempts = 0
			while True:
				number_attempts+=1


				if cLen == 1:
					#bPoint = np.random.uniform(low=0,high=1)
					bPoint = np.random.rand()
					newChrLengths = [bPoint,1-bPoint]
					bRates=np.array([beta])
					fRates=np.array([0.0])
					eventRates = bRates*1.0

				else:

					#break rates

					bRates = np.array(currentChrLengths)*beta
					


					cIdxs =list(range(cLen))
					choices = [[(a,b) for b in cIdxs] for a in cIdxs]
					choices = [item for sublist in choices for item in sublist]

					fRates = np.array( [0 if i==j else 1/((1 - abs(currentChrLengths[i]-currentChrLengths[j]))**cLen) for (i,j) in choices])*alpha

					bTotal=np.sum(bRates)
					fTotal = np.sum(fRates)
					eventRates = np.array([bTotal,fTotal])
					eventWeights = np.array([bTotal/(bTotal+fTotal),fTotal/(bTotal+fTotal)])

					event = rand_choice_nb(np.array([0,1]),eventWeights)

					if event == 0:
						weights = bRates/sum(bRates)
						choiceIdx = rand_choice_nb(np.array(list(range(cLen))),bRates)

						oldChrLengths = currentChrLengths[:]
						old = oldChrLengths.pop(choiceIdx)
						bPoint = np.random.rand()
						bPoint = bPoint * old
						
						newChrLengths = oldChrLengths + [bPoint,old-bPoint]

					if event == 1:

						weights = fRates/sum(fRates)
						pairIdx = rand_choice_nb( np.array( list(range(len(choices)))), weights)
						leftIdx,rightIdx = choices[pairIdx]
						left = currentChrLengths[leftIdx]
						right = currentChrLengths[rightIdx]

						newChrLengths =  [val for idx,val in enumerate(currentChrLengths) if idx not in [leftIdx,rightIdx]]
						newChrLengths = newChrLengths + [(left+right)]



				kCurrent = len(currentChrLengths)
				optCurrent = np.prod(np.array([1-math.exp(-1/kCurrent) for x in range(kCurrent)]))
				obsCurrent = np.prod(np.array([1-math.exp(-x) for x in currentChrLengths]))
				relCurrent = obsCurrent/optCurrent

				kNew = len(newChrLengths)
				optNew = np.prod(np.array([1-math.exp(-1/kNew) for x in range(kNew)]))
				obsNew = np.prod(np.array([1 - math.exp(-x) for x in newChrLengths]))
				relNew = obsNew/optNew

				if relNew >= relCurrent:
					break
				else:
					ranDraw = np.random.rand()
					if ranDraw <= relNew/relCurrent:
						break



		for itt2, val in enumerate(currentChrLengths): #puts the 'current' in the list so first state is in there
			resChrs[record_idx][itt2]=val

		totalRate = np.sum(eventRates)
		expTime = np.array([np.random.exponential(1/totalRate) for x in range(number_attempts)])
		expTime = np.sum(expTime)
		resTime[record_idx]=expTime
		
	return(resChrs,resTime)    
