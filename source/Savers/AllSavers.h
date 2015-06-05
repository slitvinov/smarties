/*
 *  AllSavers.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Saver.h"
#include <string>
#include <iomanip>

#include "../QApproximators/MultiTable.h"

class RewardSaver : public Saver
{
public:
    using Saver::Saver;

	void exec()
	{
		(*file) << master->getTotR() << endl;
		file->flush();
	}
};

class StateSaver : public Saver
{
private:
	MultiTable* Q;
	
public:
	using Saver::Saver;
	
	void setQ(MultiTable* nQ)
	{
		Q = nQ;
	}
	
	void exec()
	{
		ofstream& out(*file);
		map<long int, double>& data = Q->getData();
		
		_info("Saving all the states... ");
		
		for (map<long int, double>::iterator it = data.begin(); it != data.end(); it++)
		{
			State s = decode(master->sInfo, it->first / 10);
			
			out << s.vals[4] << ", " << s.vals[5] << ", " << it->second << endl;
		}
		out << endl;
		out.flush();
		
		_info("Done\n");
	}
};





