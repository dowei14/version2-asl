#include "dnf.h"
#include <math.h>
#include <iostream>

DNF::DNF(){
	stimSize = 0;
}

DNF::~DNF(){
}

void DNF::setup(int _size, double _tau, double _h, double _beta){
	size = _size;
	tau = _tau;
	h = _h;
	beta = _beta;
	for (int i=0;i<size;i++){
		activation.push_back(0.0 + h);
		output.push_back(sigmoid(activation[i],beta));
	}
}

double DNF::sigmoid(double x, double beta){
	return (1.0 / (1.0 + exp(-beta * x)));
}


double DNF::gauss(int pos, int mu, double sigma){
	if (sigma==0) {
		if (pos == mu) return 1.0;
		else return 0.0;
	} else {
		return exp(-0.5 * pow((pos-mu),2) / pow(sigma,2));
	}
}

void DNF::addStim(int pos, double sigma){
	GaussStimulus tmp;
	tmp.pos = pos;
	tmp.sigma = sigma;
	stimuli.push_back(tmp);
	stimSize++;
}

void DNF::setAmplitudes(std::vector<double> amplitudes){
	for (unsigned int i=0; i < amplitudes.size(); i++){
		stimuli[i].amplitude = amplitudes[i];
	}
}

std::vector<double> DNF::getSumStims(){
	std::vector<double> stimVec;
	int stimID = 0;
	for (int stimID=0;stimID<stimSize;stimID++){
		for (int i=0; i<size; i++){
			double value = stimuli[stimID].amplitude * gauss(i,stimuli[stimID].pos, stimuli[stimID].sigma);
			if (stimID==0) stimVec.push_back(value);
			else stimVec[i] += value;
		}
	}
	return stimVec;
}

void DNF::step(){
	std::vector<double> stimVec = getSumStims();

	// lateral step	
	lateralInteraction.fullSum = 0.0;
	for(std::vector<double>::iterator it = stimVec.begin(); it != stimVec.end(); ++it) lateralInteraction.fullSum += *it;

	// convolution
	int extendedSize = size + (size-lateralInteraction.kernelRangeRight + 1) + lateralInteraction.kernelRangeLeft;
	std::vector<double> convOut;
	for ( int i = 0; i < extendedSize; i++ ){
		convOut.push_back(0.0);
		for ( int j = 0; j < size; j++ ){
			int index;
			if (i-j >=0) index = i-j;
			else index = size + i - j;
		    convOut[i] += stimVec[index] * lateralInteraction.kernel[j];    // convolve: multiply and accumulate
		}
	}
	
	for ( int i = 0; i < size; i++ ) {
		lateralInteraction.output[i] = convOut[lateralInteraction.kernelRangeRight+i] + lateralInteraction.amplitudeGlobal * lateralInteraction.fullSum;
	}
	
	
	// step
	
	for (int i=0; i<size; i++){
		activation[i] = activation[i] + 1.0 / tau * (- activation[i] + h + (stimVec[i] + lateralInteraction.output[i]));
		output[i] = sigmoid(activation[i],beta);
	} 	
}

void DNF::setupLateral(double _sigmaExc, double _amplitudeExc, double _sigmaInh, double _amplitudeInh, double _amplitudeGlobal, double _cutoffFactor){
	lateralInteraction.sigmaExc 		= _sigmaExc;
	lateralInteraction.amplitudeExc 	= _amplitudeExc;
	lateralInteraction.sigmaInh 		= _sigmaInh;
	lateralInteraction.amplitudeInh 	= _amplitudeInh;			
	lateralInteraction.amplitudeGlobal	= _amplitudeGlobal;
	lateralInteraction.cutoffFactor 	= _cutoffFactor;

	for (int i=0; i<size; i++) lateralInteraction.output.push_back(0.0);

	
	bool useExc;
	if (lateralInteraction.amplitudeExc != 0) useExc = true;
	else useExc = false;
	
	bool useInh;
	if (lateralInteraction.amplitudeInh != 0) useInh = true;
	else useExc = false;

	double kernelRange;
	if (useExc && useInh) {
		if (lateralInteraction.sigmaExc > lateralInteraction.sigmaInh) kernelRange = lateralInteraction.sigmaExc;
		else kernelRange = lateralInteraction.sigmaInh;
	} else if (useExc) kernelRange = lateralInteraction.sigmaExc;
	else if (useInh) kernelRange = lateralInteraction.sigmaInh;
	else kernelRange = 0.0;
		
	kernelRange *= lateralInteraction.cutoffFactor;
	

	if ( ceil(kernelRange) < floor(((double)size-1.0)/2.0) ) lateralInteraction.kernelRangeLeft = ceil(kernelRange);
	else lateralInteraction.kernelRangeLeft = floor(((double)size-1.0)/2.0);

	if ( ceil(kernelRange) < ceil(((double)size-1.0)/2.0) ) lateralInteraction.kernelRangeRight = ceil(kernelRange);
	else lateralInteraction.kernelRangeRight = ceil(((double)size-1.0)/2.0);



	lateralInteraction.fullSum = 0.0;

	std::vector<double> gaussNormExc;
	std::vector<double> gaussNormInh;		
	for (int i=-lateralInteraction.kernelRangeLeft;i<=lateralInteraction.kernelRangeRight;i++){
		gaussNormExc.push_back(gauss(i,0.0,lateralInteraction.sigmaExc));
		gaussNormInh.push_back(gauss(i,0.0,lateralInteraction.sigmaInh));		
	}
	//normalize
	double sumGaussExc;
	double sumGaussInh;
	
	for(std::vector<double>::iterator it = gaussNormExc.begin(); it != gaussNormExc.end(); ++it) sumGaussExc += *it;
	for(std::vector<double>::iterator it = gaussNormInh.begin(); it != gaussNormInh.end(); ++it) sumGaussInh += *it;

	for(std::vector<double>::iterator it = gaussNormExc.begin(); it != gaussNormExc.end(); ++it) *it /= sumGaussExc;
	for(std::vector<double>::iterator it = gaussNormInh.begin(); it != gaussNormInh.end(); ++it) *it /= sumGaussInh;	
	
	for (unsigned int i=0;i<gaussNormExc.size();i++){
		lateralInteraction.kernel.push_back(lateralInteraction.amplitudeExc * gaussNormExc[i] - lateralInteraction.amplitudeInh * gaussNormInh[i]);
	}
	
	//std::cout<<kernelRange<<" "<<kernelRangeLeft<<" "<<kernelRangeRight<<std::endl;
/*	TODO:
    % initialization
    function obj = init(obj)
        obj.extIndex = [obj.size(2) - obj.kernelRangeRight + 1 : obj.size(2), 1 : obj.size(2), 1 : obj.kernelRangeLeft];      
    end
*/
}

