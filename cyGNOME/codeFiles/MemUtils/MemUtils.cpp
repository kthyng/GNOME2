/*
 *  MemUtils.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "Earl.h"
#include "TypeDefs.h"
#include <iostream>
#include "macdef.prefix"
#include "MemUtils.h"



/////////// UNIVERSAL MEMORY UTILS

long _handleCount = 0;
static long masterPointerCount = 0;
static OSErr memoryError = 0;
static Ptr *masterPointers = 0;
static Handle freeMasterPointers[10];



void  DeleteAllHandles()
{
	delete[] masterPointers;
}

Ptr  NewPtr(long size)
{
	memoryError = 0;
	
	Ptr p;
	
	try {
		p = new char[size+4];
	}
	catch(...) {
		memoryError = -1; 
		return 0; 
	}
	*((long *)p) = size;
	
	p += 4;
	memset(p, 0, size);
	
	_handleCount++;
	
	return(p);
	
}

OSErr  InitAllHandles()
{
	long kNumToAllocate = 6000; // was 1000 , JLM 6/8/10
	long i;
	
	if (!(masterPointers = (Ptr *) NewPtr(kNumToAllocate * sizeof(Ptr)))) return -1;
	
	for (i = 0 ; i < kNumToAllocate ; i++) masterPointers[i] = 0;
	for (i = 0 ; i < 10 ; i++) freeMasterPointers[i] = &masterPointers[i];
	masterPointerCount = kNumToAllocate;
	
	return 0;
}

Ptr  NewPtrClear(long size)
{
	return  NewPtr(size);
}

long  GetPtrSize(Ptr p)
{
	return *((long *)(p - 4));
}

void  SetPtrSize(Ptr p, Size newSize)
{
	Ptr p2 = 0;
	memoryError = 0;
	
	try {
		long oldSize = *((long*)(p - 4));
		p2 = new char[newSize+4];
		if(oldSize < newSize)
			memmove(p2+4, p, oldSize);
		else
			memmove(p2+4, p, newSize);
		delete[] p;
	}
	catch(...) {
		memoryError = -1;
		return;
	}
	
	*((long *)p2) = newSize;
	
	p = p2+4;
	
}


void  DisposePtr(Ptr p)
{
	delete[] p;
	_handleCount--;
}

void  DisposPtr(Ptr p)
{
	 DisposePtr(p);
}

Handle  NewHandle(long size)
{
	long i, j;
	Ptr p;
	Handle h = 0;
	
	// look for a free space
	// freeMasterPointers holds the easy places to look
	for (i = 0 ; i < 10 ; i++)
		if (freeMasterPointers[i] != 0)
		{ h = freeMasterPointers[i]; break; } // we found an easy one
	
	if (!h) {
		// did not find a free place for our handle
		// we need to reset the freeMasterPointers
		for (i = 0, j = 0 ; i < masterPointerCount ; i++)
			if (masterPointers[i] == 0) 
			{	// we found a free unused place
				if (!h) h = &masterPointers[i];
				if (j < 10)
					freeMasterPointers[j++] = &masterPointers[i];
				else
					break;// we have found all ten
			}
		if (!h) {
			// JLM 6/8/10, the InitWindowsHandles below will loose track of hte previously allocated "Handles".
			// as a work around to help avoid this, I've increase the number allocated in InitWindowsHandles 
			 InitAllHandles();
			h = &masterPointers[0];
			// code goes here, JLM 3/30/99, why was this code was commented out ?
			// Note: This code is still in TAT
			/*
			 masterPointers = (Ptr *)Win_SetPtrSize(masterPointers, (masterPointerCount + 1000) * sizeof(Ptr));
			 if (memoryError) return 0;
			 for (i = 0 ; i < 10 ; i++)
			 freeMasterPointers[i] = &masterPointers[masterPointerCount + i];
			 h = &masterPointers[masterPointerCount];
			 masterPointerCount += 1000;
			 */
		}
	}
	
	// we have found all ten
	if (!(p =  NewPtr(size))) return 0;// unable to allocate
	
	(*h) = p;// record the pointer in the MAC-like "Handle"
	
	for (i = 0 ; i < 10 ; i++)
		if (freeMasterPointers[i] == h)
		{
			freeMasterPointers[i] = 0;// mark this place as no longer free
			break;
		}
	
	return h;
}


Handle  NewHandleClear(long size)
{
	return  NewHandle(size);
}

Handle  TempNewHandle(long size, LONGPTR err)
{
	Handle h =  NewHandle(size);
	
	(*err) = memoryError;
	
	return h;
}

Handle  RecoverHandle(Ptr p)
{
	long i;
	
	memoryError = 0;
	
	for (i = 0 ; i < masterPointerCount ; i++)
		if (masterPointers[i] == p)
			return &masterPointers[i];
	
	memoryError = -1;
	
	return 0;
}


void  HLock(Handle h)
{
	return; // all handles are always locked in Windows
}

void  MyHLock(Handle h)
{
	 HLock(h);
}
void  HUnlock(Handle h)
{
	return; // all handles are always locked in Windows
}

void  SetHandleSize(Handle h, long newSize)
{
	 SetPtrSize(*h, newSize);
}


Handle  MySetHandleSize(Handle h, long newSize)
{
	 SetHandleSize(h, newSize);
	
	return h;
}


long  GetHandleSize(Handle h)
{
	return  GetPtrSize(*h);
}
OSErr  HandToHand(HANDLEPTR hp)
{
	long size =  GetHandleSize(*hp);
	Handle h2;
	
	h2 =  NewHandle(size);
	if (!h2) return -1;
	
	memcpy(*h2, **hp, size);
	// CopyMemory(h2, *hp, size);
	
	*hp = h2;
	
	return 0;
}


void  DisposeHandleReally(Handle h)
{
	short i;
	
	 DisposePtr(*h);
	
	*h = 0;
	
	for (i = 0 ; i < 10 ; i++)
		if (freeMasterPointers[i] == 0)
		{ freeMasterPointers[i] = h; break; }
}

void  DisposeHandle2(HANDLEPTR hp)
{
	 DisposeHandleReally(*hp);
	*hp = 0;
}

void  DisposeHandle(Handle h) {
	DisposeHandle(h);
}

short  ZeroHandleError(Handle h)
{
	if (h == 0)
#ifdef IBM
#ifndef NO_GUI
		MessageBox(hMainWnd, "Dereference of NULL handle.", 0, 0);
#endif
#else
	std::cerr << "Dereference of a NULL handle.";
#endif
	
	return 0;
}

void  BlockMove(VOIDPTR sourcePtr, VOIDPTR destPtr, long count)
{
	memmove(destPtr, sourcePtr, count);
}

void  MyBlockMove(VOIDPTR sourcePtr, VOIDPTR destPtr, long count)
{
	memmove(destPtr, sourcePtr, count);
}

#define THE_UPPER_BOUND 2147483647L

long  MaxBlock (void)
// We'll do this until we find a better way,
{
	static long N = THE_UPPER_BOUND/2;
	
	try {
		char *p;
		int m = N;
		p = new char[m];
		delete[] p;
		N = THE_UPPER_BOUND/2;
		return m;
	}
	
	catch(...) {
		N /= 2;
		return MaxBlock();
	}
	
}


OSErr  MemError (void)
{
	return memoryError;
}

///// SYSTEM  STUFF /////////////////////////

OSErr DeleteHandleItem(CHARH h, long i, long size)
{
	long total =  GetHandleSize((Handle)h),
	elements = total / size;
	
	 BlockMove(&INDEXH(h, (i + 1) * size),
			   &INDEXH(h, i * size),
			   (elements - (i + 1)) * size);
	
	 SetHandleSize((Handle)h, total - size);
	
	return  MemError();
}



long MyTempMaxMem() {
	return  MaxBlock();
}

Handle MyNewHandleTemp(long size)
{
	return  NewHandle(size);
	
}

