/*
 *  ShioTimeValue_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ShioTimeValue_c__
#define __ShioTimeValue_c__

#include "Shio.h"
#include "OSSMTimeValue_c.h"

#define MAXNUMSHIOYEARS  20
#define MAXSTATIONNAMELEN  128
#define kMaxKeyedLineLength	1024

#ifndef pyGNOME
#include "TMover.h"
#else
#include "Mover_c.h"
#define TMover Mover_c
#endif

//enum { WIZ_POPUP = 1, WIZ_UNITS , WIZ_EDIT, WIZ_BMP, WIZ_HELPBUTTON };

typedef struct
{
	short year;// 1998, etc
	YEARDATAHDL yearDataHdl;
} ShioYearInfo;

typedef struct
{
	Seconds time;
	double speedInKnots;
	short type;	// 0 -> MinBeforeFlood, 1 -> MaxFlood, 2 -> MinBeforeEbb, 3 -> MaxEbb
} EbbFloodData,*EbbFloodDataP,**EbbFloodDataH;

typedef struct
{
	Seconds time;
	double height;
	short type;	// 0 -> Low Tide, 1 -> High Tide
} HighLowData,*HighLowDataP,**HighLowDataH;

YEARDATAHDL GetYearData(short year);

class ShioTimeValue_c : virtual public OSSMTimeValue_c {

protected:

	// instance variables
	char fStationName[MAXSTATIONNAMELEN];
	char fStationType;
	double fLatitude;
	double fLongitude;
	CONSTITUENT2 fConstituent;
	HEIGHTOFFSET fHeightOffset;
	CURRENTOFFSET fCurrentOffset;
	//
	Boolean fHighLowValuesOpen; // for the list
	Boolean fEbbFloodValuesOpen; // for the list
	EbbFloodDataH fEbbFloodDataHdl;	// values to show on list for tidal currents
	HighLowDataH fHighLowDataHdl;	// values to show on list for tidal heights
	
	OSErr		GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float *** val);
	OSErr 		GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,DATA * val);
	OSErr 		GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,short * val);
	OSErr 		GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float * val);
	OSErr 		GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,double * val);
	OSErr		GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr		GetTimeChange (long a, long b, Seconds *dt);
	
	void 		ProgrammerError(char* routine);
	void 		InitInstanceVariables(void);
	
	long 		I_SHIOHIGHLOWS(void);
	long 		I_SHIOEBBFLOODS(void);
	
public:						
							ShioTimeValue_c() { fEbbFloodDataHdl = 0; fHighLowDataHdl = 0;}
							ShioTimeValue_c (TMover *theOwner);
							ShioTimeValue_c (TMover *theOwner,TimeValuePairH tvals);
	virtual ClassID 		GetClassID () { return TYPE_SHIOTIMEVALUES; }
	virtual Boolean			IAm(ClassID id) { if(id==TYPE_SHIOTIMEVALUES) return TRUE; return OSSMTimeValue_c::IAm(id); }
	virtual OSErr			ReadTimeValues (char *path);
	virtual long			GetNumEbbFloodValues ();	
	virtual long			GetNumHighLowValues ();
	virtual OSErr			GetTimeValue (Seconds time, VelocityRec *value);
	virtual WorldPoint		GetRefWorldPoint (void);
	
	virtual	double			GetDeriv (Seconds t1, double val1, Seconds t2, double val2, Seconds theTime);
	virtual	OSErr			GetConvertedHeightValue(Seconds forTime, VelocityRec *value);
	virtual	OSErr			GetProgressiveWaveValue(Seconds forTime, VelocityRec *value);
	OSErr 					GetLocationInTideCycle(short *ebbFloodType, float *fraction);
	virtual OSErr			InitTimeFunc ();

	
	
};


#undef TMover
#endif