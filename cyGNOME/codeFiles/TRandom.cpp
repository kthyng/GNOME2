
#include "Cross.h"

TRandom *sharedRMover;


#ifdef MAC
#ifdef MPW
#pragma SEGMENT TRANDOM
#endif
#endif

///////////////////////////////////////////////////////////////////////////

TRandom::TRandom (TMap *owner, char *name) : TMover (owner, name)
{
	fDiffusionCoefficient = 100000; //  cm**2/sec 
	memset(&fOptimize,0,sizeof(fOptimize));
	SetClassName (name);
	fUncertaintyFactor = 2;		// default uncertainty mult-factor
	bUseDepthDependent = false;
}


long TRandom::GetListLength()
{
	long count = 1;
	
	if (bOpen) {
		count += 2;
		if(model->IsUncertain())count++;
	}
	
	return count;
}

ListItem TRandom::GetNthListItem(long n, short indent, short *style, char *text)
{
	ListItem item = { this, 0, indent, 0 };
	char valStr[32];
	
	if (n == 0) {
		item.index = I_RANDOMNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Random: \"%s\"", className);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	n -= 1;
	item.indent++;
	
	if (bOpen) {
		if (n == 0) {
			item.index = I_RANDOMACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		
		n -= 1;
		
		if (n == 0) {
			item.index = I_RANDOMAREA;
			StringWithoutTrailingZeros(valStr,fDiffusionCoefficient,0);
			sprintf(text, "%s cm**2/sec", valStr);
			
			return item;
		}
		
		n -= 1;
		
		if(model->IsUncertain())
		{
			if (n == 0) {
				item.index = I_RANDOMUFACTOR;
				StringWithoutTrailingZeros(valStr, fUncertaintyFactor,0);
				sprintf(text, "Uncertainty factor: %s", valStr);
				
				return item;
			}
	
			n -= 1;
		}
		
	}
	
	item.owner = 0;
	
	return item;
}

Boolean TRandom::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_RANDOMNAME: bOpen = !bOpen; return TRUE;
			case I_RANDOMACTIVE: bActive = !bActive; 
				model->NewDirtNotification(); return TRUE;
		}
	
	if (doubleClick)
		RandomSettingsDialog(this, this -> moverMap);
	
	// do other click operations...
	
	return FALSE;
}

Boolean TRandom::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_RANDOMNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
			}
			break;
		default:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
	}
	
	return TMover::FunctionEnabled(item, buttonID);
}

OSErr TRandom::SettingsItem(ListItem item)
{
	switch (item.index) {
		default:
			return RandomSettingsDialog(this, this -> moverMap);
	}
	
	return 0;
}

OSErr TRandom::DeleteItem(ListItem item)
{
	if (item.index == I_RANDOMNAME)
		return moverMap -> DropMover (this);
	
	return 0;
}

OSErr TRandom::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char ourName[kMaxNameLen];
	Boolean useDepthDependent;
	// see if the message is of concern to us
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val = 0;
		OSErr  err;
		err = message->GetParameterAsDouble("coverage",&val); // old style
		if(err) err = message->GetParameterAsDouble("Coefficient",&val);
		if(!err)
		{	
			if(val >= 0)// do we have any other  max or min limits ?
			{
				this->fDiffusionCoefficient = val;
				model->NewDirtNotification();// tell model about dirt
			}
		}
		///
		err = message->GetParameterAsDouble("Uncertaintyfactor",&val);
		if(!err)
		{	
			if(val >= 1.0)// do we have any other max or min limits ?
			{
				this->fUncertaintyFactor = val;
				model->NewDirtNotification();// tell model about dirt
			}
		}
		err = message->GetParameterAsBoolean("DepthDependent",&useDepthDependent);
		if(!err)
		{	
			this->bUseDepthDependent = useDepthDependent;
			//model->NewDirtNotification();// tell model about dirt
		}
		///
		
	}
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TMover::CheckAndPassOnMessage(message);
}

/////////////////////////////////////////////////

OSErr TRandom::PrepareForModelStep()
{
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6.*(fDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.uncertaintyValue = sqrt(fUncertaintyFactor*6.*(fDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	return noErr;
}

void TRandom::ModelStepIsDone()
{
	memset(&fOptimize,0,sizeof(fOptimize));
}

/////////////////////////////////////////////////

WorldPoint3D TRandom::GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double		dLong, dLat;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	float rand1,rand2;
	double 	diffusionCoefficient;

	//if (deltaPoint.z > 0) return deltaPoint;	// only use for surface LEs ?

	if (bUseDepthDependent)
	{
		float depth;
		double localDiffusionCoefficient, factor;
		TVectorMap* vMap = GetNthVectorMap(0);	// get first vector map
		if (vMap) depth = vMap->DepthAtPoint(refPoint);
		// logD = 1+exp(1-1/.1H) 
		if (depth==0)	// couldn't find the point in dagtree, maybe a different default?
			factor = 1;
		else
			factor = 1 + exp(1 - 1/(.1*depth));
		if (depth>20)
		//localDiffusionCoefficient = pow(10,factor);
		localDiffusionCoefficient = pow(10.,factor);
		else
		localDiffusionCoefficient = 0;
		this -> fOptimize.value =  sqrt(6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		/*if (depth<20)
		{
			localDiffusionCoefficient = 0;
			this -> fOptimize.value =  sqrt(6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
			this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		}
		else
		{
			localDiffusionCoefficient = 0;
			this -> fOptimize.value =  sqrt(6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
			this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		}*/
		// MoverMap->GetGrid();	// mover map will likely be universal map
		// need to get the bathymetry then set diffusion based on > 20 O(1000), < 20 O(100)
		// figure out where LE is, interpolate to get depth (units?)
	}
	if(!this->fOptimize.isOptimizedForStep && !bUseDepthDependent)  
	{
		this -> fOptimize.value =  sqrt(6.*(fDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(fDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
	}

	if (leType == UNCERTAINTY_LE)
		diffusionCoefficient = this -> fOptimize.uncertaintyValue;
	else
		diffusionCoefficient = this -> fOptimize.value;

	if(this -> fOptimize.isFirstStep)
	{
		GetRandomVectorInUnitCircle(&rand1,&rand2);
	}
	else
	{
		rand1 = GetRandomFloat(-1.0, 1.0);
		rand2 = GetRandomFloat(-1.0, 1.0);
	}
	
	dLong = (rand1 * diffusionCoefficient )/ LongToLatRatio3 (refPoint.pLat);
	dLat  = rand2 * diffusionCoefficient;
	 
	// code goes here
	// note: could add code to make it a circle the first step

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;

	return deltaPoint;
}

//#define TRandom_FileVersion 1
#define TRandom_FileVersion 2
OSErr TRandom::Write(BFPB *bfpb)
{
	long version = TRandom_FileVersion;
	ClassID id = GetClassID ();
	OSErr err = 0;
	
	if (err = TMover::Write(bfpb)) return err;
	
	StartReadWriteSequence("TRandom::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	if (err = WriteMacValue(bfpb, fDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, fUncertaintyFactor)) return err;
	
	if (err = WriteMacValue(bfpb, bUseDepthDependent)) return err;

	return 0;
}

OSErr TRandom::Read(BFPB *bfpb) 
{
	long version;
	ClassID id;
	OSErr err = 0;
	
	if (err = TMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TRandom::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TRandom::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TRandom_FileVersion || version < 1) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &fDiffusionCoefficient)) return err;
	if (err = ReadMacValue(bfpb, &fUncertaintyFactor)) return err;
	if (version>1)
		if (err = ReadMacValue(bfpb, &bUseDepthDependent)) return err;

	return 0;
}

OSErr M28Init (DialogPtr dialog, VOIDPTR data)
// new random diffusion dialog init
{
	SetDialogItemHandle(dialog, M28HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M28FROST1, (Handle)FrameEmbossed);

	mysetitext(dialog, M28NAME, sharedRMover -> className);
	MySelectDialogItemText (dialog, M28NAME, 0, 100);

	SetButton(dialog, M28ACTIVE, sharedRMover -> bActive);

	Float2EditText(dialog, M28DIFFUSION, sharedRMover->fDiffusionCoefficient, 0);

	Float2EditText(dialog, M28UFACTOR, sharedRMover->fUncertaintyFactor, 0);
	
	return 0;
}

short M28Click (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
// old random diffusion dialog
{
	char	name [kMaxNameLen];
	double uncertaintyFactor; 
	
	switch (itemNum) {
		case M28OK:
		
			// uncertaintyFactor enforce >= 1.0
			uncertaintyFactor = EditText2Float(dialog, M28UFACTOR);
			if(uncertaintyFactor <1.0)
			{
				printError("The uncertainty factor must be >= 1.0");
				MySelectDialogItemText (dialog, M28UFACTOR, 0, 100);
				break;
			}
			mygetitext(dialog, M28NAME, name, kMaxNameLen - 1);		// get the mover's nameStr
			sharedRMover -> SetClassName (name);
			sharedRMover -> SetActive (GetButton(dialog, M28ACTIVE));
			sharedRMover -> fDiffusionCoefficient = EditText2Float(dialog, M28DIFFUSION);
			sharedRMover -> fUncertaintyFactor = uncertaintyFactor;

			return M28OK;

		case M28CANCEL: return M28CANCEL;
		
		case M28ACTIVE:
			ToggleButton(dialog, M28ACTIVE);
			break;
			
		case M28DIFFUSION:
			CheckNumberTextItem(dialog, itemNum, false); //  don't allow decimals
			break;

		case M28UFACTOR:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;

	}
	
	return 0;
}

OSErr RandomSettingsDialog(TRandom *mover, TMap *owner)
{
	short item;
	TRandom *newMover = 0;
	OSErr err = 0;
	
	if (!mover) {
		newMover = new TRandom(owner, "Diffusion");
		if (!newMover)
			{ TechError("RandomSettingsDialog()", "new TRandom()", 0); return -1; }
		
		if (err = newMover->InitMover()) { delete newMover; return err; }
		
		sharedRMover = newMover;
	}
	else
		sharedRMover = mover;
	
	item = MyModalDialog(M28, mapWindow, 0, M28Init, M28Click);
	
	if (item == M28OK) model->NewDirtNotification();

	if (newMover) {
		if (item == M28OK) {
			if (err = owner->AddMover(newMover, 0))
				{ newMover->Dispose(); delete newMover; return -1; }
		}
		else {
			newMover->Dispose();
			delete newMover;
		}
	}
	
	return 0;
}

