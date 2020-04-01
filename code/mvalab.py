# -*- coding: utf-8 -*-
#
# Lecture des fichiers "télécom"
# Sylvain Lobry 2015
# JM Nicolas 2015#
#
# Septembre 2017 : on introduit mat2imz
#   sauvegarde en 8 bits ou en float. Pas d'autre choix
#   possibilité de générer un fichier .hdr en mettant une option
#
# Janvier 2018 : introduction de dat2mat pour le projet biomass
#               : introduction de cos2mat (uniquement les complex short)
#
# Aout 2018 : possibilité de lire un fichier url
#
##########################################################################
#
# Aout 2018 : introduction d'un test de version pour urllib
#
##########################################################################
#
#  Sauvegarde au fomat Télécom : mat2imz
#
##########################################################################
#
# Affichage d'images "visusar"
#
#
#########################################################################
u"""
lecture, écriture et affichage d'images à forte dynamique, réeles ou complexes (radar)
"""

MVALABVERSION=u"V2.1  Version du 5 février 2019"

#############################################################################

import sys  # pour la version de python
import os.path

import numpy as npy

import scipy.fftpack as spyfftpack
import scipy.io as spio


import struct
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import math

import urllib
from urllib import *

globalparamnotebook=0

##############################################################################
#
def version():
    return MVALABVERSION
##############################################################################
#############################
def notebook(*therest):
    u'''
    Sans argument : modifie certains affichages pour les notebooks

    Avec argument :

        ==0 : affichage normal

        ==1 : affichage pour notebook
    '''
    global globalparamnotebook
    globalparamnotebook=1
    if len(therest)==1 :
        globalparamnotebook=therest[0]

#General IO functions

################################################
#
def matlab2imz(fichier, namechamp):
    try :
        data = spio.loadmat(fichier)
    except :
        print(u'Erreur dans le nom du .mat')
        return 0, 0, 0
    try :
        master = data[namechamp]
        ncol=npy.size(master,1)
        nlig=npy.size(master,0)
        return master, ncol, nlig, 1
    except :
        print(u'Erreur dans le nom du champ du .mat')
        return 0, 0, 0



############################################################################
# Septembre 2017 : une liste de paramètres sont requis
#
def imz2matbase(namima, listeparam):
#
# on rajoute un test sur l'existence même du fichier
#

    print('imz2mat, appel imz2matbase : version Septembre 2017')

    try :
        ftest=open(namima);
    except IOError:
        legx=namima+': est un fichier non ouvrable'
        print(legx)
        print(u'Echec à l\'appel de imz2matbase')
        return 0,0,0,0
    ftest.close()

    nparam=npy.size(listeparam)
    if(nparam!=7):
        print(u'Echec appel imz2matbase : nombre erroné de paramètres (%d'%nparam+' au lieu de 7')
        return 0,0,0,0
#    for iut in range(nparam) :
#        print(listeparam[iut])

##############################################################################
# Septembre 2017
#



    nblok=    listeparam[0]*listeparam[1]*  listeparam[5]  *  (1+listeparam[6])
    ncan=0
    return( _readImage( namima, listeparam[0], listeparam[1], listeparam[2], 1, listeparam[3], listeparam[4], listeparam[5], listeparam[6], nblok, ncan))


def imz2mat(imgName,ncan=0):
    """
    lecture d'images plutot radar
    Formats Telecom et .dat
    argument 1 : nom du fichier image (ou de l'url d'un fichier image)
    argument 2 (facultatif) : si multicanal, renvoie uniquement le canal indiqué (ATTENTION : numérotation à prtir de 0)
    """

#
# on rajoute un test dur l'existence même du fichier
#

    print(u'imz2mat : version janvier 2018.  Fichier à ouvrir : %s'%imgName)

    if imgName.startswith('http')==True :
        print('Nom compatible url')
        return(urlimz2mat(imgName, ncan))


    try :
        ftest=open(imgName);
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        print(u'Echec à l\'appel de imz2mat')
        return 0,0,0,0

    ftest.close()

##############################################################################
# Septembre 2017
#
    if imgName.endswith('.dim') :
        ncolZ, nligZ, nplantotZ, nzzZ = dimimabase(imgName )
        offsetZ, nbBytesZ, typeZ, komplexZ, radarZ, endianZ, namima = dimimadim(imgName)
        print(dimimabase(imgName))
        print(dimimadim(imgName))
        nblok=ncolZ*nligZ*nbBytesZ*(1+komplexZ)
        return( _readImage( namima, ncolZ, nligZ, nplantotZ, nzzZ, offsetZ, endianZ+typeZ, nbBytesZ, komplexZ, nblok, ncan))

##############################################################################
# Janvier 2018 : données .dat de l'ONERA (projet Biomass)
#
    if imgName.endswith('.dat') :
        return( dat2mat(imgName ))  # voir à la fin de ce fichier, avant les outils de visu

##############################################################################
# Mai 2018 : données .cos du DLR
#
    if imgName.endswith('.cos') :
        return( cos2mat(imgName ))  # voir à la fin de ce fichier, avant les outils de visu

##############################################################################
##############################################################################
# Septembre 2017
#
        # faire la meme chose avec les .hdr !!!
###############################################################################


    ncolZ, nligZ, nplantotZ, nzzZ = dimimabase(imgName )

    if(nplantotZ==1):
        print("Dans ximaread : image monocanal")

    if(nplantotZ>0):
        print("Dans ximaread : lecture du canal "+"%d"%ncan+'/'+'%d'%nplantotZ)


    """ Reads a file in a xima format. """
    if imgName.endswith('.ima'):
        print("image en .ima")
        return imaread(imgName,ncan)
    elif imgName.endswith('.IMA'):
        print("image en .IMA")
        return imaread(imgName,ncan)
    elif imgName.endswith('.imw'):
        print("image en .imw")
        return imwread(imgName,ncan)
    elif imgName.endswith('.IMW'):
        print("image en .IMW")
        return imwread(imgName,ncan)
    elif imgName.endswith('.iml'):
        print("image en .iml")
        return imlread(imgName,ncan)
    elif imgName.endswith('.IML'):
        print("image en .IML")
        return imlread(imgName,ncan)
    elif imgName.endswith('.rvb'):
        print("image en .rvb")
        return imaread(imgName, 1)  ########### TODO
    elif imgName.endswith('.cxs'):
        print("image en .cxs")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.cxb'):
        print("image en .cxb")
        return cxbread(imgName,ncan)
    elif imgName.endswith('.cxbtivo'):
        print("image en .cxbtivo")
        return cxbread(imgName,ncan)
    elif imgName.endswith('.cxbadts'):
        print("image en .cxbadts")
        return cxbread(imgName,ncan)
    elif imgName.endswith('.CXS'):
        print("image en .CXS")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.cxstivo'):
        print("image en .cxstivo")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.CXSTIVO'):
        print("image en .CXSTIVO")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.cxsadts'):
        print("image en .cxsadts")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.CXSADTS'):
        print("image en .CXSADTS")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.imf'):
        print("image en .imf")
        return imfread(imgName,ncan)
    elif imgName.endswith('.IMF'):
        print("image en .IMF")
        return imfread(imgName,ncan)
    elif imgName.endswith('.imd'):
        print("image en .imd")
        return imdread(imgName,ncan)
    elif imgName.endswith('.IMD'):
        print("image en .IMD")
        return imdread(imgName,ncan)
    elif imgName.endswith('.cxf'):
        print("image en .cxf")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.CXF'):
        print("image en .CXF")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.cxftivo'):
        print("image en .cxftivo")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.CXFTIVO'):
        print("image en .CXFTIVO")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.cxfadts'):
        print("image en .cxfadts")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.CXFADTS'):
        print("image en .CXFADTS")
        return cxfread(imgName,ncan)
    else:
#        raise Exception("Format not currently supported.")
        print("Format non pris en compte actuellement");
        return 0,0,0,0,0
#IO operations for ima format. both IMA and ima
def imaread(imgName,ncan):
    """ Reads a *ima file. ImgName can be with or without extension. """
    if imgName.endswith('.ima'):
        print("image en .ima")
        extension='.ima'
    if imgName.endswith('.IMA'):
        print("image en .IMA")
        extension='.IMA'
    if imgName.endswith('.rvb'):
        print("image en .rvb")
        extension='.rvb'
    imgName = os.path.splitext(imgName)[0]
    return _imaread(imgName, extension,ncan)
#IO operations for imw format.
def imwread(imgName,ncan):
    """ Reads a *.imw file. ImgName can be with or without extension. """
    if imgName.endswith('.imw'):
        print("image en .imw")
        extension='.imw'
    if imgName.endswith('.IMW'):
        print("image en .IMW")
        extension='.IMW'
    imgName = os.path.splitext(imgName)[0]
    return _imwread(imgName, extension,ncan)
#IO operations for iml format.
def imlread(imgName,ncan):
    """ Reads a *.iml file. ImgName can be with or without extension. """
    if imgName.endswith('.iml'):
        print("image en .iml")
        extension='.iml'
    if imgName.endswith('.IML'):
        print("image en .IML")
        extension='.IML'
    imgName = os.path.splitext(imgName)[0]
    return _imlread(imgName, extension,ncan)
#IO operations for cxb format.
def cxbread(imgName,ncan):
    """ Reads a *.imw file. ImgName can be with or without extension. """
    if imgName.endswith('.cxb'):
        print("image en .cxb")
        extension='.cxb'
    if imgName.endswith('.cxbtivo'):
        print("image en .cxbtivo")
        extension='.cxbtivo'
    if imgName.endswith('.cxbadts'):
        print("image en .cxbadts")
        extension='.cxbadts'
    imgName = os.path.splitext(imgName)[0]
    return _cxbread(imgName, extension,ncan)
#IO operations for cxs format.
def cxsread(imgName,ncan):
    """ Reads a *.cxs file. ImgName can be with or without extension. """
    if imgName.endswith('.cxs'):
        print("image en .cxs")
        extension='.cxs'
    if imgName.endswith('.CXS'):
        print("image en .CXS")
        extension='.CXS'
    if imgName.endswith('.cxstivo'):
        print("image en .cxstivo")
        extension='.cxstivo'
    if imgName.endswith('.CXSTIVO'):
        print("image en .CXSTIVO")
        extension='.CXSTIVO'
    if imgName.endswith('.cxsadts'):
        print("image en .cxsadts")
        extension='.cxsadts'
    if imgName.endswith('.CXSADTS'):
        print("image en .CXSADTS")
        extension='.CXSADTS'
    imgName = os.path.splitext(imgName)[0]
    return _cxsread(imgName, extension,ncan)
#IO operations for imf format.
def imfread(imgName,ncan):
    """ Reads a *.imf file. ImgName can be with or without extension. """
    if imgName.endswith('.imf'):
        print("image en .imf")
        extension='.imf'
    if imgName.endswith('.IMF'):
        print("image en .IMF")
        extension='.IMF'
    imgName = os.path.splitext(imgName)[0]
    return _imfread(imgName, extension,ncan)
#IO operations for imd format.
def imdread(imgName,ncan):
    """ Reads a *.imf file. ImgName can be with or without extension. """
    if imgName.endswith('.imd'):
        print("image en .imd")
        extension='.imd'
    if imgName.endswith('.IMD'):
        print("image en .IMD")
        extension='.IMD'
    imgName = os.path.splitext(imgName)[0]
    return _imdread(imgName, extension,ncan)
#IO operations for cxf format.
def cxfread(imgName,ncan):
    """ Reads a *.cxf file. ImgName can be with or without extension. """
    if imgName.endswith('.cxf'):
        print("image en .cxf")
        extension='.cxf'
    if imgName.endswith('.CXF'):
        print("image en .CXF")
        extension='.CXF'
    if imgName.endswith('.cxftivo'):
        print("image en .cxftivo")
        extension='.cxftivo'
    if imgName.endswith('.CXFTIVO'):
        print("image en .CXFTIVO")
        extension='.CXFTIVO'
    if imgName.endswith('.cxfadts'):
        print("image en .cxfadts")
        extension='.cxfadts'
    if imgName.endswith('.CXFADTS'):
        print("image en .CXFADTS")
        extension='.CXFADTS'
    imgName = os.path.splitext(imgName)[0]
    return _cxfread(imgName, extension,ncan)
#Internal functions.
def _imaread(imgName, extension, ncan):
    """ Reads a *.ima file. imgName should come with no extension. """
        #

    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en .ima ',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=1
    offset=0
    type='B'
    komplex=0
    nblok=w*h
    indien = '>' #en minuscule

    if extension == '.ima':
        indien='>'
    if extension == '.IMA':
        indien='<'
    if extension == '.rvb':
        komplex=999
        nbBytes=1
    if nktemps>0:
        offset, nbBytes, type, komplex, radar, indienZ, namerien = _readDimparamZV2(imgName + '.dim')
        if indienZ != 'Z':
            indien=indienZ
    nblok=nblok*nbBytes*(1+komplex) ######## ATTENTION : ne marchera que pour des tivoli
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, indien+type, nbBytes, komplex, nblok, ncan)



def _imwread(imgName, extension, ncan):
    """ Reads a *.imw file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en unsigned short',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=2
    offset=0
    if extension == '.imw':
        typeA='H'
        endian='>'
    if extension == '.IMW':
        typeA='H'
        endian='<'
    komplex=0
    nblok=w*h*2
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _imlread(imgName, extension, ncan):
    """ Reads a *.iml file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en int',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=4
    offset=0
    if extension == '.iml':
        typeA='i'
        endian='>'
    if extension == '.IML':
        typeA='i'
        endian='<'
    komplex=0
    nblok=w*h*4
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _imfread(imgName, extension, ncan):
    """ Reads a *.imf file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en float',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=4
    offset=0
    if extension == '.imf':
        typeA='f'
        endian='>'
    if extension == '.IMF':
        typeA='f'
        endian='<'
    komplex=0
    nblok=w*h*4
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _imdread(imgName, extension, ncan):
    """ Reads a *.imd file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en double',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=8
    offset=0
    if extension == '.imd':
        typeA='d'
        endian='>'
    if extension == '.IMD':
        typeA='d'
        endian='<'
    komplex=0
    nblok=w*h*4
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _cxbread(imgName, extension, ncan):
    """ Reads a *.cxb file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en complex signed char',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=1
    offset=0
    nblok=w*h*2
    if extension == '.cxb':
        typeA='b'
        endian='>'
        komplex=1
    if extension == '.cxbtivo':
        typeA='b'
        endian='>'
        komplex=2
    if extension == '.cxbadts':
        typeA='b'
        endian='>'
        komplex=3
    if nktemps>0:
        offset, nbBytes, type, komplex, radar, indien, namerien = _readDimparamZV2(imgName + '.dim')
    if radar==1:
        komplex=11
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, type, nbBytes, komplex, nblok, ncan)
def _cxsread(imgName, extension, ncan):
    """ Reads a *.cxs file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en complex signed short',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=2
    offset=0
    nblok=w*h*4
    if extension == '.cxs':
        typeA='h'
        endian='>'
        komplex=1
    if extension == '.CXS':
        typeA='h'
        endian='<'
        komplex=1
    if extension == '.cxstivo':
        typeA='h'
        endian='>'
        komplex=2
    if extension == '.CXSTIVO':
        typeA='h'
        endian='<'
        komplex=2
    if extension == '.cxsadts':
        typeA='h'
        endian='>'
        komplex=3
    if extension == '.CXSADTS':
        typeA='h'
        endian='<'
        komplex=3
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)

def _cxfread(imgName, extension, ncan):
    """ Reads a *.cxf file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en float',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=4
    offset=0
    if extension == '.cxf':
        typeA='f'
        endian='>'
        komplex=1
    if extension == '.CXF':
        typeA='f'
        endian='<'
        komplex=1
    if extension == '.cxftivo':
        typeA='f'
        endian='>'
        komplex=2
    if extension == '.CXFTIVO':
        typeA='f'
        endian='<'
        komplex=2
    if extension == '.cxfadts':
        typeA='f'
        endian='>'
        komplex=3
    if extension == '.CXFADTS':
        typeA='f'
        endian='<'
        komplex=3
    nblok=w*h*8
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)

def _readDimZ(dimFile):
    """ Reads a *.dim file and return width and height. """
    try :
        f=open(dimFile);
    except IOError:
        legx=dimFile+': est un fichier non ouvrable'
        print(legx)
        return 0,0,0,0
    else:
        tmp = f.readline().split()
        w = int(tmp[0])
        h = int(tmp[1])
        nk=1
        nktemps=0
# sert de code retour si le fichier .dim n'a que 2 valeurs
        if len(tmp)>2:
            print('Fichier .dim version longue (lecture 3eme parametre) ')
            nk = int(tmp[2]);
        if len(tmp)>3:
            print('Fichier .dim version longue (lecture 4eme parametre)')
            nktemps = int(tmp[3]);
        return w, h, nk, nktemps

def _readDimparamZV2(dimFile):
    """ Reads a *.dim file and return width and height. """
    offsetZ=0
    nbBytesZ=1
    typeZ='B'
    komplexZ=0
    radarZ=0
    endianZ='Z'
    namima=""
    with open(dimFile) as f:
        tmpK = f.readline()
        while tmpK!='':
            tmp = tmpK.split()
            print(tmp[0], tmp[1])
            if tmp[0]=="-offset":
                # print("Lecture de l''offset  ",tmp[1])
                offsetZ=tmp[1]
            if tmp[0]=="-radar":
                # print("Lecture du radar  ",tmp[1])
                if tmp[1]=="ERS":
                    radarZ=1
            if tmp[0]=="-image":
                print("Le .dim contient le nom de l\'image : ",tmp[1])
                namima=tmp[1]
            if tmp[0]=="-bo":  # bug corrigé !!!
                # print("endian ?? :   ",tmp[1])
                if tmp[1]=="SUN":
                    endianZ='>'
                if tmp[1]=="DEC":
                    endianZ='<'
            if tmp[0]=="-type":
                # print("Lecture du type  ",tmp[1])
                if tmp[1]=="U8":
                    nbBytesZ=1
                    typeZ='B'
                if tmp[1]=="U16":
                    nbBytesZ=2
                    typeZ='H'
                if tmp[1]=="S16":
                    nbBytesZ=2
                    typeZ='h'
                if tmp[1]=="S32":
                    nbBytesZ=4
                    typeZ='i'
                if tmp[1]=="U32":
                    nbBytesZ=4
                    typeZ='I'
                if tmp[1]=="FLOAT":
                    nbBytesZ=4
                    typeZ='f'
                if tmp[1]=="DOUBLE":
                    nbBytesZ=8
                    typeZ='d'
                if tmp[1]=="C8":
                    nbBytesZ=1
                    komplexZ=1
                    typeZ='b'
                if tmp[1]=="CS8":
                    nbBytesZ=1
                    komplexZ=1
                    typeZ='b'
                if tmp[1]=="CS8TIVO":
                    nbBytesZ=1
                    komplexZ=2
                    typeZ='b'
                if tmp[1]=="CS8ADTS":
                    nbBytesZ=1
                    komplexZ=3
                    typeZ='b'
                if tmp[1]=="CS16":
                    nbBytesZ=2
                    komplexZ=1
                    typeZ='h'
                if tmp[1]=="CS16TIVO":
                    nbBytesZ=2
                    komplexZ=2
                    typeZ='h'
                if tmp[1]=="CS16ADTS":
                    nbBytesZ=2
                    komplexZ=3
                    typeZ='h'
                if tmp[1]=="C32TIVO":
                    nbBytesZ=4
                    komplexZ=2
                    typeZ='f'
                if tmp[1]=="C32ADTS":
                    nbBytesZ=4
                    komplexZ=3
                    typeZ='f'
                if tmp[1]=="CFLOAT":
                    nbBytesZ=4
                    komplexZ=1
                    typeZ='f'

            tmpK = f.readline()


        return offsetZ, nbBytesZ, typeZ, komplexZ, radarZ, endianZ, namima


def _readImage(imgName, w, h, nkparam, nktemps, offset, typeA, nbBytes, komplex, nblok, ncan):
    print("lecture de ", imgName,' en quelconque', w, h, nkparam, ' offset ', offset, typeA, nbBytes,' complex',komplex,'blocksize',nblok)
    if ncan>0:
        print('lecture specifique du canal %d'%ncan)
#    print('parametre ncan : %d'%ncan)
    """ Reads an image coded in any binary format. """

    tagRNSAT=0
    nk=nkparam

    if  nkparam<0 :
        print(u'Fichier RNSat : procédure en test')
        nk=-nkparam
        tagRNSAT=1

    try :
        f=open(imgName,'rb');
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        return 0,0,0,0,0
    else:
        f.seek(offset,0)
        if komplex==999:
            imgligne=npy.empty([3*w])
            img = npy.empty([h, w, 3])
            for i in range(0, h):
#                if i%100==0:
#                    print(u'Ligne lue %d'%i)
                record=f.read(nbBytes*3*w)
                imgligne = npy.ndarray( 3*w, typeA, record)
                img[i, 0:h, 0]= imgligne[0:3*h:3]
                img[i, 0:h, 1]= imgligne[1:3*h:3]
                img[i, 0:h, 2]= imgligne[2:3*h:3]

#            imgligne=npy.empty([3*w])
#            for i in range(0, h):
#                for j in range(0, 3*w):
#                    imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
#                for j in range(0, w):
#                    img[i, j, 0]=imgligne[3*j]/255.
#                    img[i, j, 1]=imgligne[3*j+1]/255.
#                    img[i, j, 2]=imgligne[3*j+2]/255.
            return img,w,h,nk,nktemps


        if nk>1:
            tag3=1

        if nk==1 :
            tag3=0
            nkmin=0
            nkmax=1

        if  ncan==0:
            tag3=0
            nkmin=0
            nkmax=nk
            if nkmax>1:
                tag3=1
#            print(nkmin)
#            print(nkmax)
#            print(tag3)

        if ncan>0:
            if ncan>nk:
                ncan=nk
            tag3=2
            nkmin=ncan-1
            nkmax=ncan


        if(ncan>1):
            f.seek(nblok*(ncan-1))

        if tag3==1:
            nkmin=0
            nkmax=nk
            if komplex==0:
                imgtot = npy.empty([h, w, nk])
            else:
                imgtot = npy.zeros([h, w, nk])+ 1j * npy.zeros([h, w, nk])

            if tagRNSAT==1 :
                if komplex==0:
                    imgtotstep = npy.empty([h*w*nk])
                if komplex==1:
                    imgtotstep = npy.zeros([h*w*nk]) + 1j * npy.zeros([h*w*nk])
                iutrnsat=0
                iblocRNSAT=h*w


        if komplex==0:
            img = npy.empty([h, w])

        if komplex==1 or komplex==2 or komplex==11:
            #print(h,w)
            imgligne=npy.empty([2*w])
            img = npy.zeros([h, w])+ 1j * npy.zeros([h, w])

        if komplex==3:
            imgampli = npy.empty([h, w])
            imgphase = npy.empty([h, w])
            img = npy.zeros([h, w])+ 1j * npy.zeros([h, w])

#        print('Boucle de lecture entre %d'%nkmin+' et %d'%nkmax+'   sur %d'%nk+' canaux'+'  (tag3=%d'%tag3+')')
        if nk>1:
            print('Boucle de lecture entre %d'%nkmin+' et %d'%nkmax+'   sur %d'%nk+' canaux')

#        print('Verif typeA = %s'%typeA )


        for nkz in range(nkmin,nkmax):
            if tag3==1 or tag3==2:
                print('Lecture du canal %d'%(nkz+1)+'/%d'%nk)
#            else:
#                print('Lecture monocanal  %d'%(nkz+1)+'/%d'%nk)

#############
            if komplex==0:
                print(u'Données réelles. Nouvelle version de imz2mat  '+'%s'%typeA)
                record=npy.zeros(nbBytes*w, dtype=npy.byte() )
                for i in range(0, h):
                    record = f.read(nbBytes*w)
                    img[i, :] = npy.ndarray( w, typeA, record)
#                    for j in range(0, w):
#                        img[i, j] = struct.unpack( typeA, record[nbBytes*j:nbBytes*j+nbBytes])[0]

############  komplex > 0  : trois cas

            if komplex==1 or komplex==11:   # cas des complexes standards :
                                            # partie réelle puis partie imaginaire
#                for i in range(0, h):
#                    for j in range(0, 2*w):
#                        imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
#                    for j in range(0, w):
#                        img[i, j] = imgligne[2*j]+imgligne[2*j+1]*1j

#>>>>>>>>>>>>  a incorporer !!
#   imgligneZ =  numpy.ndarray( 2*nfencboucle, '<h', record[4*vigdebcol:4*(vigdebcol+nfencboucle)])

                print(u'Données complexes (standard). Nouvelle version de imz2mat  '+'%s'%typeA)
                record=npy.zeros(nbBytes*w*2, dtype=npy.byte() )
                for i in range(0, h):
                    record = f.read(nbBytes*w*2)
                    imgligne =  npy.ndarray( 2*w, typeA, record)
                    img[i,:]=imgligne[0:2*w:2]+1j*imgligne[1:2*w:2]
#                    for j in range(0, w):
#                        img[i, j] = imgligne[2*j]+imgligne[2*j+1]*1j

            if komplex==11:
                valmoyR = npy.mean(img.real)
                valmoyI = npy.mean(img.imag)
                img = img.real-valmoyR+(img.imag-valmoyI)*1j
## komplex==2
            if komplex==2:  # d'abord la partie réelle, puis la partir imaginaire
                for i in range(0, h):
                    for j in range(0, w):
                        imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
                    for j in range(0, w):
                        img[i, j] =  imgligne[j]
                for i in range(0, h):
                    for j in range(0, w):
                        imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
                    for j in range(0, w):
                        img[i, j] =  img[i, j] + imgligne[j]*1j
## komplex==3
            if komplex==3: # d'abord l'amplitude, puis la phase
                imgampli = npy.empty([h, w])
                imgphase = npy.empty([h, w])
                for i in range(0, h):
                    for j in range(0, w):
                        imgampli[i, j] = struct.unpack( typeA, f.read(nbBytes))[0]
                for i in range(0, h):
                    for j in range(0, w):
                        imgphase[i, j] = struct.unpack( typeA, f.read(nbBytes))[0]
                for i in range(0, h):
                    for j in range(0, w):
                        img[i, j] = imgampli[i, j]*(cos(imgphase[i, j])+sin(imgphase[i, j])*1j)
###################################################################################


            if tag3==1 and tagRNSAT==0:
                imgtot[:,:,nkz]=img[:,:]


            if tag3==1 and tagRNSAT==1:  #horrible verrue
                for iutloop in range(iblocRNSAT) :
                    jbase=iutloop%w
                    ibase=int(iutloop/w)
                    jspe=(iutrnsat%nk)*iblocRNSAT
                    ispe=int(iutrnsat/nk)
                    imgtotstep[ispe+jspe]=img[ibase,jbase]
                    iutrnsat=iutrnsat+1


        if tagRNSAT==1:  # je ne m'en suis pas sorti avec les reshape
            ispe=w*h
            for iut in range(nk):
                isk=iut*iblocRNSAT
                for jut in range(h) :
                    imgtot[jut,:,iut]=imgtotstep[jut*w+isk:(jut+1)*w+isk]

        if tag3==0 or tag3==2:
            return img,w,h,nk,nktemps
        else:
            print('retour tableau 3-D (%dx%dx%d)'%(w,h,nk))
            return imgtot,w,h,nk,nktemps
#
#############################################"
def dat2mat(imgName):
#
# on rajoute un test dur l'existence même du fichier
#

    print('dat2mat : version Janvier 2018')

    try :
        fin=open(imgName,'rb');
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        print(u'Echec à l\'appel de dat2mat')
        return 0,0,0,0


    firm = fin.read(4)
    nlig = struct.unpack("h",fin.read(2))[0]
    ncol = struct.unpack("h",fin.read(2))[0]

#
    firm=npy.zeros(8*ncol, dtype=npy.byte() )
    imgcxs = npy.empty([nlig-1, ncol], dtype=npy.complex64())

# on elimine la première ligne : cf la doc !!
    firm = fin.read(8*ncol)

    for iut in range(nlig-1):
        firm = fin.read(8*ncol)
        imgligne =  npy.ndarray( 2*ncol, 'f', firm)
        imgcxs[iut,:]=imgligne[0:2*ncol:2]+1j*imgligne[1:2*ncol:2]


    return imgcxs, ncol, nlig-1, 1, 1


def dimdat(imgName):
#
# on rajoute un test dur l'existence même du fichier
#

    print('dimdat : version Janvier 2018')

    try :
        fin=open(imgName,'rb');
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        print(u'Echec à l\'appel de dimdat')
        return 0,0,0,0


    firm = fin.read(4)
    nlig = struct.unpack("h",fin.read(2))[0]
    ncol = struct.unpack("h",fin.read(2))[0]

    return ncol, nlig-1, 1, 1
#############################################"
def cos2mat(imgName):
#
# on rajoute un test dur l'existence même du fichier
#

    print('cos2mat : version Juin 2018')

    try :
        fin=open(imgName,'rb');
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        print(u'Echec à l\'appel de cos2mat')
        return 0,0,0,0


    ibib=struct.unpack(">i",fin.read(4))[0]
    irsri=struct.unpack(">i",fin.read(4))[0]
    irs=struct.unpack(">i",fin.read(4))[0]
    ias=struct.unpack(">i",fin.read(4))[0]
    ibi=struct.unpack(">i",fin.read(4))[0]
    irtnb=struct.unpack(">i",fin.read(4))[0]
    itnl=struct.unpack(">i",fin.read(4))[0]

    nlig = struct.unpack(">i",fin.read(4))[0]
    ncoltot = int(irtnb/4)
    ncol=ncoltot-2
    nlig=ias

    print(u'Image Terrasar-X  format DLR.  ncol=%d  nlig=%d'%(ncol,nlig))

#
    firm=npy.zeros(4*ncoltot, dtype=npy.byte() )
    imgcxs = npy.empty([nlig, ncol], dtype=npy.complex64())

#
#    firm=npy.zeros(8*ncol, dtype=npy.byte() )
#    imgcxs = npy.empty([nlig-1, ncol], dtype=npy.complex64())
#
## on elimine les deux premières lignes : cf la doc !!
    fin.seek(0)
    firm = fin.read(4*ncoltot)
    firm = fin.read(4*ncoltot)
    firm = fin.read(4*ncoltot)
    firm = fin.read(4*ncoltot)
#
    for iut in range(nlig):
        firm = fin.read(4*ncoltot)
        imgligne =  npy.ndarray( 2*ncoltot, '>h', firm)
        imgcxs[iut,:]=imgligne[4:2*ncoltot:2]+1j*imgligne[5:2*ncoltot:2]


#    return imgcxs, ncol, nlig, 1, 1
    return  imgcxs, ncol, nlig, 1, 1
#
def dimcos(imgName):
#
# on rajoute un test dur l'existence même du fichier
#

    print('cos2mat : version Juin 2018')

    try :
        fin=open(imgName,'rb');
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        print(u'Echec à l\'appel de cos2mat')
        return 0,0,0,0


    ibib=struct.unpack(">i",fin.read(4))[0]
    irsri=struct.unpack(">i",fin.read(4))[0]
    irs=struct.unpack(">i",fin.read(4))[0]
    ias=struct.unpack(">i",fin.read(4))[0]
    ibi=struct.unpack(">i",fin.read(4))[0]
    irtnb=struct.unpack(">i",fin.read(4))[0]
    itnl=struct.unpack(">i",fin.read(4))[0]

    nlig = struct.unpack(">i",fin.read(4))[0]
    ncoltot = irtnb/4
    ncol=ncoltot-2
    nlig=ias

    print(u'Image Terrasar-X  format DLR.  ncol=%d  nlig=%d'%(ncol,nlig))

    return  ncol, nlig, 1, 1


##########################################################
# AOUT 2018
# Lecture d'une partie des images Télécom Paristech vua url
#
#########################################################

def urlimz2mat(nomurl, ncanselect) :

    if nomurl.startswith('http')==True :
        print('Nom compatible url')
    else:
        print(u'%s >> Ce nom n''est pas celui d''une url'%nomurl)
        return 0,0,0,0,0

    versionpython=sys.version
    tagcmplx=1

    ##################### tests de tail : seulement certains fichiers Telecom !!
    # attention : tagcmplx est égal à 1 par défaut
    #   est égal à 2 si complexe
    #   est égal à 0 si couleur rvb

    if nomurl.endswith('.ima') :
        print('Unsigned bytes')
        nbBytes=1
        typeA='B'
        endian='>'

    if nomurl.endswith('.rvb') :
        print('Image couleur, Unsigned bytes')
        nbBytes=3
        typeA='B'
        endian='>'
        tagcmplx=0

    if nomurl.endswith('.imw') :
        print('Unsigned short, Fichiers Unix')
        nbBytes=2
        typeA='H'
        endian='>'
    if nomurl.endswith('.IMW') :
        print('Unsigned short,Fichiers PC')
        nbBytes=2
        typeA='H'
        endian='<'

    if nomurl.endswith('.imf') :
        print('Float, Fichiers Unix')
        nbBytes=4
        typeA='f'
        endian='>'
    if nomurl.endswith('.IMF') :
        print('Float ,Fichiers PC')
        nbBytes=4
        typeA='f'
        endian='<'


    if nomurl.endswith('.cxb') :
        print('Complex char, Fichiers Unix')
        tagcmplx=2
        nbBytes=1
        typeA='b'
        endian='>'

    if nomurl.endswith('.cxs') :
        print('Complex short, Fichiers Unix')
        tagcmplx=2
        nbBytes=2
        typeA='h'
        endian='>'

    if nomurl.endswith('.CXS') :
        print('Complex short, Fichiers PC')
        tagcmplx=2
        nbBytes=2
        typeA='h'
        endian='<'

    if nomurl.endswith('.cxf') :
        print('Complex float, Fichiers Unix')
        tagcmplx=2
        nbBytes=4
        typeA='f'
        endian='>'
    if nomurl.endswith('.CXF') :
        print('Complex float, Fichiers PC')
        tagcmplx=2
        nbBytes=4
        typeA='f'
        endian='<'


    namedim = os.path.splitext(nomurl)[0]+'.dim'
    print("INFO - reading header/dim : "+namedim)

    if versionpython[0] == '2' :
        try :
            dataficdim=urllib.urlopen( namedim)
        except urllib.error.URLError as e:
            print(u'Erreur à la lecture de %s :'%namedim,  e.reason)
            return 0,0,0,0,0

        dataficdim=urllib.urlopen( namedim)


    if versionpython[0] == '3' :
        try :
            dataficdim=urllib.request.urlopen( namedim)
        except urllib.error.URLError as e:
            print(u'Erreur à la lecture de %s :'%namedim,  e.reason)
            return 0,0,0,0,0

        dataficdim=urllib.request.urlopen( namedim)



    datadim= dataficdim.readline().split()
    #        tmp = f.readline().split()
    largeur = int(datadim[0])
    hauteur = int(datadim[1])
    nkan=1
    nn=len(datadim)
    if nn > 2 :   # le .dim a une première ligne "complète"
        nplan=int(datadim[2])
        print(u'Multitemporal image %d data'%nplan)
        if nomurl.endswith('.ima') :
            print('Fichiers Unix')
            endian='>'
        if nomurl.endswith('.IMA') :
            print('Fichiers PC')
            endian='<'
        nkan=int(datadim[2])
        nbBytes=0
        tagcmplx=1
        while len(datadim)>0 :
            datadim= dataficdim.readline().split()
#            print(datadim)
#            if len(datadim)>0 and str(datadim[0]) == "b'-type'" :
            if versionpython[0] == '2' :
                if len(datadim)>0 and str(datadim[0]) == '-type' :
                    if str(datadim[1]) == 'CFLOAT' :
                        tagcmplx=2
                        nbBytes=4
                        typeA='f'
                        print('Image  complex float')
                    if str(datadim[1]) == 'CS16' :
                        tagcmplx=2
                        nbBytes=2
                        typeA='h'
                        print('Image  complex short')
                    if str(datadim[1]) == 'U16' :
                        tagcmplx=2
                        nbBytes=2
                        typeA='h'
                        print('Image Unsigned short')
            if versionpython[0] == '3' :
                if len(datadim)>0 and str(datadim[0]) == "b'-type'" :
                    if str(datadim[1]) == "b'CS16'" :
                        tagcmplx=2
                        nbBytes=2
                        typeA='h'
                        print('Image  complex short')
                if len(datadim)>0 and str(datadim[0]) == "b'-type'" :
                    if str(datadim[1]) == "b'CFLOAT'" :
                        tagcmplx=2
                        nbBytes=4
                        typeA='f'
                        print('Image  complex float')
                    if str(datadim[1]) == "b'U16'" :
                        tagcmplx=2
                        nbBytes=2
                        typeA='h'
                        print('Image Unsigned short')


    print('lecture .dim OK -> largeur:%d hauteur:%d profondeur:%d'%(largeur,hauteur,nkan))

    if tagcmplx==0 :
        imgligne=npy.empty([tagcmplx*3*largeur])
        img = npy.zeros([hauteur, largeur, 3])
    else :
        imgligne=npy.empty([tagcmplx*nbBytes*largeur])
        print(u'Debug %d %d %d'%(tagcmplx,nbBytes,largeur))

    if tagcmplx==1 :
        if nkan==1:
            img = npy.zeros([hauteur, largeur])
        if nkan>1:
            if ncanselect==0 :
                img = npy.zeros([hauteur, largeur, nkan])
            else :
                img = npy.zeros([hauteur, largeur])

    if tagcmplx==2 :
        if nkan==1:
            img = npy.zeros([hauteur, largeur])+ 1j * npy.zeros([hauteur, largeur])
        if nkan>1:
            if ncanselect==0 :
                img = npy.zeros([hauteur, largeur, nkan])+ 1j * npy.zeros([hauteur, largeur, nkan])
            else :
                img = npy.zeros([hauteur, largeur])+ 1j * npy.zeros([hauteur, largeur])

#####################################
    if versionpython[0] == '2' :
        try :
            dataficima=urllib.urlopen(nomurl )
        except urllib.error.URLError as e:
            print(u'Erreur à la lecture de %s :'%nomurl,  e.reason)
            return 0,0,0,0,0
        dataficima=urllib.urlopen(nomurl )


    if versionpython[0] == '3' :
        try :
            dataficima=urllib.request.urlopen(nomurl )
        except urllib.error.URLError as e:
            print(u'Erreur à la lecture de %s :'%nomurl,  e.reason)
            return 0,0,0,0,0
        dataficima=urllib.request.urlopen(nomurl )


###
#    if tagcouleur==1 : #TOTO
#        print(u'Images rvb non traitées pour le moment')
#        return 0,0,0,0,0




# ????         return img, largeur, hauteur, 1, 1
    #record=npy.zeros(nbBytes*largeur*tagcmplx, dtype=npy.byte() )
    print(nbBytes,largeur, endian+typeA)
    if nkan==1 :
        if tagcmplx==0 :# rvb en 3 couleurs
            for i in range(0, hauteur):
                record = dataficima.read(nbBytes*largeur)
                imgligne =  npy.ndarray( nbBytes*largeur, endian+typeA, record)
                img[i,:,0]=imgligne[0:nbBytes*largeur:3]
                img[i,:,1]=imgligne[1:nbBytes*largeur:3]
                img[i,:,2]=imgligne[2:nbBytes*largeur:3]

        if tagcmplx==1 :
            for i in range(0, hauteur):
                record = dataficima.read(nbBytes*largeur)
                imgligne =  npy.ndarray( largeur, endian+typeA, record)
                img[i,:]=imgligne[0:largeur]

        if tagcmplx==2 :
            for i in range(0, hauteur):
                record = dataficima.read(nbBytes*largeur*2)
                imgligne =  npy.ndarray( 2*largeur, endian+typeA, record)
                img[i,:]=imgligne[0:2*largeur:2]+1j*imgligne[1:2*largeur:2]

    if nkan>1 and ncanselect==0 :
        if tagcmplx==1 :
            for nk in range(nkan):
                for i in range(0, hauteur):
                    record = dataficima.read(nbBytes*largeur)
                    imgligne =  npy.ndarray( largeur, endian+typeA, record)
                    img[i,:,nk]=imgligne[0:largeur]

        if tagcmplx==2 :
            for nk in range(nkan):
                for i in range(0, hauteur):
                    record = dataficima.read(nbBytes*largeur*2)
                    imgligne =  npy.ndarray( 2*largeur, endian+typeA, record)
                    img[i,:,nk]=imgligne[0:2*largeur:2]+1j*imgligne[1:2*largeur:2]

    if nkan>1 and ncanselect>0 :
        print(u'Sélection du canal %d/%d'%(ncanselect,nkan))
        if tagcmplx==1 :
            for nk in range(ncanselect-1):
                for i in range(0, hauteur):
                    record = dataficima.read(nbBytes*largeur)

            for i in range(0, hauteur):
                record = dataficima.read(nbBytes*largeur)
                imgligne =  npy.ndarray( largeur, endian+typeA, record)
                img[i,:]=imgligne[0:largeur]

        if tagcmplx==2 :
            for nk in range(ncanselect-1):
                for i in range(0, hauteur):
                    record = dataficima.read(nbBytes*largeur*2)

            for i in range(0, hauteur):
                record = dataficima.read(nbBytes*largeur*2)
                imgligne =  npy.ndarray( 2*largeur, endian+typeA, record)
                img[i,:]=imgligne[0:2*largeur:2]+1j*imgligne[1:2*largeur:2]

    if ncanselect>0 :
        nkan=1

#
# il faut éventuellement recentrer les .cxb
    if tagcmplx==2 and nbBytes==1 :
        img=img-npy.mean(img)

    return  img, largeur, hauteur, nkan, 1


#######################################
def dimimabase(imgName):
    """
    Renvoie les 4 paramètres d'une image
    Paramètre en entrée : le nom de l'image avec son extension (soit extensions telecom, soit .cos --Terrasar-X--, soit .dat --Ramses--)
    Sortie : une liste avec les 4 PARAMÈTRES  nombre de colonnes,  nombre de lignes, nombre de canaux, nombre de plans
    """

    if imgName.endswith('.cos') :  # DLR
        return dimcos(imgName)

    if imgName.endswith('.dat') :  # ONERA
        return dimdat(imgName)


    if imgName.endswith('.dim') :
        w, h, nk, nktemps  = _readDimZ(imgName)
    else :
        imgName = os.path.splitext(imgName)[0]
        w, h, nk, nktemps  = _readDimZ(imgName + '.dim')
#    print('n colonnes '+'%d'%w+'  nlignes '+'%d'%h+' ncanaux '+'%d'%nk)
    return w, h, nk, nktemps

#
#######################################################################################
#######################################################################################
    #  FIN DES LECTURES
#######################################################################################
#######################################################################################

######################################################
######################################################
# SEPTEMBRE 2017
#

typecode='<f'  # < : little endian
hdrcode='byte order = 0'  # pour .hdr d'IDL/ENVI
imacode='-byteorder = 0'  # pour .dim (mesure conservatoire)


def mat2imz( tabimage, nomimage, *therest):

    """
    Procedure pour ecrire un tableau dans un fichier au format TelecomParisTech
    Le tableau sera archivé en :
        .ima si tableau 8 bits
        .IMF sinon
        .CXF si complexe
    Si le tableau est à 3 dimensions (pile TIVOLI), l'archivage se fera en .IMA
    Exemple d'appel :
    mat2imz( montableau2d, 'MaSortie')
    Pour avoir aussi  le fichier .hdr d'IDL
    mat2imz( montableau2d, 'MaSortie', 'idl')
    """




    nomdim=nomimage+'.dim'

    taghdr=0
    testchar=0

    if(len(therest)==1):
        if therest[0]=="idl" :
            taghdr=1

    ndim=npy.ndim(tabimage)
    if(ndim<2):
        print('mat2imz demande un tableau 2D ou 3D')
        return
    if(ndim>3):
        print('mat2imz demande un tableau 2D ou 3D')
        return


    nlig=npy.size(tabimage,0)
    ncol=npy.size(tabimage,1)
    nplan=1  # par defaut.. pour idl
#
# Cas image 2D
#
    if ndim==2:
        fp=open(nomdim,'w')
        fp.write('%d'%ncol+'  %d'%nlig)
        fp.close()
        imode=npy.iscomplex(tabimage[0][0])
        if imode==True :
            nomimagetot=nomimage+'.CXF'
            fp=open(nomimagetot,'wb')
            for iut in range(nlig):
                for jut in range(ncol):
                    fbuff=float(tabimage.real[iut][jut])
                    record=struct.pack( typecode, fbuff)
                    fp.write(record)
                    fbuff=float(tabimage.imag[iut][jut])
                    record=struct.pack( typecode, fbuff)
                    fp.write(record)
            fp.close()

        else :
            mintab=npy.min(tabimage)
            maxtab=npy.max(tabimage)
            if(mintab>-0.0001 ) :
                if (maxtab<255.0001):
                    testchar=1
                    nomimagetot=nomimage+'.ima'
                    ucima=npy.uint8(tabimage)
                    fp=open(nomimagetot,'wb')
                    for iut in range(nlig):
                        for jut in range(ncol):
                            record=struct.pack( 'B', ucima[iut][jut])
                            fp.write(record)

                else :
                    nomimagetot=nomimage+'.IMF'
                    fp=open(nomimagetot,'wb')
                    for iut in range(nlig):
                        for jut in range(ncol):
                            fbuff=float(tabimage[iut][jut])
                            record=struct.pack( typecode, fbuff)
                            fp.write(record)
            fp.close()





    if ndim==3:
        nplan=npy.size(tabimage,2)
        imode=npy.iscomplex(tabimage[0][0][0])
        mintab=npy.min(tabimage)
        maxtab=npy.max(tabimage)
        if(mintab>-0.0001 ) :
            if (maxtab<255.0001):
                testchar=1


        fp=open(nomdim,'w')
        fp.write('%d'%ncol+'  %d'%nlig+'  %d'%nplan+'   1'+'\n')
        if imode==True :
            fp.write('-type CFLOAT')
        else :
            if testchar==0:
                fp.write('-type FLOAT')
            if testchar==1:
                fp.write('-type U8')
        fp.close()

        nomimagetot=nomimage+'.IMA'
        fp=open(nomimagetot,'wb')
        if imode==True :
            for lut in range(nplan):
                for iut in range(nlig):
                    for jut in range(ncol):
                        fbuff=float(tabimage.real[iut][jut][lut])
                        record=struct.pack( typecode, fbuff)
                        fp.write(record)
                        fbuff=float(tabimage.imag[iut][jut][lut])
                        record=struct.pack( typecode, fbuff)
                        fp.write(record)
        else :
            if(testchar==1) :
                for lut in range(nplan):
                    ucima=npy.uint8(tabimage[:,:,lut])
                    for iut in range(nlig):
                        for jut in range(ncol):
                            record=struct.pack( 'B', ucima[iut][jut])
                            fp.write(record)
            else :
                for lut in range(nplan):
                    for iut in range(nlig):
                        for jut in range(ncol):
                            fbuff=float(tabimage[iut][jut][lut])
                            record=struct.pack( typecode, fbuff)
                            fp.write(record)


        fp.close()

    if taghdr==1 :
        noffset=0
        nomhdr=nomimagetot+'.hdr'
        fp=open(nomhdr,'w')
        fp.write('ENVI \n')
        fp.write('{Fichier produit par tiilab.mat2imz (python) } \n')
        fp.write('samples = %d'%ncol+'\n')
        fp.write('lines = %d'%nlig+'\n')
        fp.write('bands = %d'%nplan+'\n')
        fp.write('header offset = %d'%noffset+'\n')
        fp.write('file type = ENVI Standard \n')
        if imode==True :
            fp.write('data type = 6 \n')
        else :
            if(testchar==1) :
                fp.write('data type = 1  \n')
            else :
                fp.write('data type = 4  \n')

        fp.write('interleave = bsq \n')
        fp.write(hdrcode+'\n')
        if imode==True :
            fp.write('complex function = Magnitude  \n')


        fp.close()




######################################################
######################################################
        #######################################################################################
#  OUTILS VISU
#######################################################################################
#######################################################################################
#
def visusarbase(tabimaparam,paramseuil,tagspe, *therest):
#
# on commence par eliminer les tableaux manifestement trop petits...
# ainsi que les tableaux reduit a la valeur 0 (entiere)
#
    # si tagspe==0 : pas de show, ni de titre
#
# Mai 2018 : *therest pour ne remplir qu'une partie de la figure
# d'où ipart, mais qui sert aussi à neutraliser les affichages
#
# Sept 2018 : zparam peut aussi être un tableau [vmin,vmax]
#
#
# therest :
#   [0] : pour les affichages
#   [1] : pour les histogrammes     if type(tabima)== str :
    if type(tabimaparam)== str :
        tabimatab=imz2mat(tabimaparam)
        tabima = tabimatab[0]
    else :
        tabima=tabimaparam
#
    ipart=0
    kparamhisto=0
    if len(therest)>0 :
        ipart=therest[0]

    if isinstance(tabima,int)==True:
        legspe='Pas de visualisation : Tableau nul'
        print(legspe)
        return

    RSI=tabima.size
    if RSI<16:
        legspe='Pas de visualisation : Tableau manifestement beaucoup trop petit (%d) pour etre une image : pas d''affichage' %(RSI)
        print(legspe)
        return 0

#    else:
#        print("visusar sans second parametre'

    R=tabima.shape
    ZZ=len(R)
    if ZZ==3:  # on traite à part les couleurs
        malegende='3 canaux couleurs (RVB)'
        print("Affichage comme image en couleur (3 canaux)")
        if tagspe > 0 :
            plt.figure()
            if ipart==1 :
                print(u'Figure réduite pour l\'image')
        plt.imshow(tabima/255.,interpolation='nearest')
        if tagspe > 0 :
            plt.title(malegende)
            plt.show()
        return 0



    if npy.isrealobj(tabima)==True:
        print("Affichage d'une image reelle")
        BB=tabima
    if npy.isrealobj(tabima)==False:
        print("Affichage d'une image complexe : on prend le module")
        BB=abs(tabima)
    valmin=npy.min(BB)
    valmax=npy.max(BB)
    valsig=npy.std(BB)
    valmoy=npy.mean(BB)
    legx='Min %.3f   Max %.3f    Moy %.3f   Ect %.3f ' %(valmin,valmax,valmoy,valsig)

#
#  par defaut visusarbase affiche avec un seuil de vmoy + kparam * vsig et avec kparam = 3
    kparam=3  # par defaut
    itagseuil=1
    seuilmin=0.  # par defaut

    if type(paramseuil) == list :
        seuilmin=paramseuil[0]
        seuilmax=paramseuil[1]
        kparam=(seuilmax-valmoy)/valsig
        if kparamhisto==0 :
            kparamhisto=kparam
        print(u'seuil min (%.1f) et seuilmax (%.1f) passés en argument. kparam = %.1f'%(seuilmin, seuilmax, kparam))
        malegende=u'Image affichée entre %.1f et %.1f  (vmoy + %.1f vsig)'%(seuilmin,seuilmax,kparam)
    else:
    # ancien cas
        zparam = paramseuil


        if zparam!=-999:
            kparam=zparam

        if kparam<0 :
            itagseuil=2
            kparam=-kparam

        if kparam>0:
            seuilmax=valmoy+kparam*valsig
            if seuilmax>valmax:
                BB[0,0]=seuilmax
            if kparamhisto==0 :
                kparamhisto=kparam

            if itagseuil==2 :
                seuilmin=valmoy-kparam*valsig

            malegende=u'Image  [%.2f, %.2f] seuil %.2f \n   valmoy (%.3f) + %.3f sigma (%.3f) ' %(seuilmin, valmax, seuilmax,valmoy,kparam,valsig)

#            seuilmax=valmax
#            malegende='Image sans seuillage'


#   Ancienne version avec seuil fixe...
# a réintroduire peut être
#    if kparam<0:
#        seuilmax=-kparam
#        malegende='Image seuillee : %.2f' %(seuilmax)
#        masque=BB<seuilmax
#        BB=BB*masque+(1-masque)*seuilmax
#
        if kparam==0:
            malegende='kparam=0 : Image sans seuillage'
            seuilmin=valmin
            seuilmax=valmax
###########" fin cas historique


    masque=BB<seuilmax
    BB=BB*masque+(1-masque)*seuilmax
    masque2=BB>seuilmin
    BB=BB*masque2+(1-masque2)*seuilmin

#        malegende='Image seuillee : valmoy (%.3f) +- %.3f sigma  (%.2f;%.2f)\n seuil %.1f' %(valmoy,kparam,seuilmin,seuilmax)


    if tagspe==0:
        print('Visusar sans plt.show, ni titre')

    if tagspe <0 :
        print('Visusar sans affichage')
        print(legx)
        return(BB)


    if len(therest)>1 :
#        print(u'Passage de kparamhisto en parametre %.3f', therest[1] )
        if therest[1] != 0. :
            kparamhisto=therest[1]


    if tagspe > -1 :
        if tagspe>0 :
            if globalparamnotebook==0 :
                fig = plt.figure()
            if globalparamnotebook>0 :
                fig = plt.figure(figsize=(12,12))
            plt.xlabel(legx)


        if ipart>0 :
            if tagspe==0:
                print(u'Affichage pas conforme (appel de l''histogramme en mode Z)')
            gs = gridspec.GridSpec( 8, 1)
            plt.subplot( gs[0:7,0] )
            plt.xticks([])
            plt.yticks([])

#        plt.imshow(BB)
        if tagspe>0 :
            plt.suptitle(malegende)
        cax=plt.imshow(BB,interpolation='nearest')

#        plt.suptitle(malegende)
#
#        if tagspe>0 :
#            plt.suptitle(malegende)
#        if ipart==2 :
#            plt.title(malegende)  # placement très subtil, avant l'éventuel histogramme
#
        plt.set_cmap('gray')
        if len(therest)>2 :
            plt.set_cmap('jet')
        if ipart>0 :
            plt.subplot( gs[7,0] )
            ntot=R[0]*R[1]
            seuilmax=valmoy+kparamhisto*valsig
            noutlier=npy.sum(tabima>seuilmax)
            fntot=float(ntot)
            fnoutlierpourcent=float(noutlier)*100./fntot

            seuilmaxhisto= seuilmax
#            if  seuilmaxhisto > valmax :
#                seuilmaxhisto = valmax   # octobre 2018 : source d'erreur pour l'opérateur
#            plt.hist(npy.reshape(tabima,ntot), bins=256, normed=True, color='wheat')
            if npy.isrealobj(tabima)==True:
                print('Histogramme des valeurs d''une image (%.3f %.3f) entre %.3f et %.3f'%( npy.min(tabima), npy.max(tabima),seuilmin,seuilmaxhisto))
#                print("Histogramme d'une image reelle")plt.hist(crop_classe0.ravel(),bins=256)
#                plt.hist(npy.reshape(tabima,ntot), range=[seuilmin,seuilmaxhisto], bins=256, normed=True, color='wheat')
                resulvoid=plt.hist(tabima.ravel(), range=[seuilmin,seuilmaxhisto], bins=256, normed=True, color='wheat')
            else:
                print(u'Histogramme des valeurs absolues d''une image complexe')
#                plt.hist(npy.reshape(npy.abs(tabima),ntot), range=[seuilmin,seuilmaxhisto], bins=256, normed=True, color='wheat')
                resulvoid=plt.hist(npy.abs(tabima.ravel()), range=[seuilmin,seuilmaxhisto], bins=256, normed=True, color='wheat')
            yyymin,yyymax=plt.ylim()
            plt.plot([seuilmax,seuilmax],[yyymin,yyymax],'k--')
            if kparamhisto > kparam and seuilmax < valmax:
                if npy.isrealobj(tabima)==True:
                    noutlierhisto=npy.sum(tabima>seuilmax)
                else:
                    noutlierhisto=npy.sum(npy.abs(tabima)>seuilmax)
                fnoutlierhistopourcent=float(noutlierhisto)*100./fntot
###                plt.xlim([0.,seuilmaxhisto])
                baratinx=u'%d pixels > %.1f (%.1f  %%) ,   %d pixels > %.1f (%.3f  %%)'%(noutlier,seuilmax,fnoutlierpourcent,noutlierhisto,seuilmax,fnoutlierhistopourcent)
            else :
                baratinx=u'[%d]    %d pixels > %.1f (%.1f  %%)'%(int(fntot), noutlier,seuilmax,fnoutlierpourcent)

            if seuilmin > 0.001 :
                noutliermin=npy.sum(tabima<seuilmin)
                if noutliermin > 0 :
                    fnoutlierminpourcent=float(noutliermin)*100./fntot
                    baratinx=u'%d pixels < %.1f (%.1f  %%)      '%(noutliermin,seuilmin,fnoutlierminpourcent)+baratinx


            plt.xlabel(baratinx)
            plt.yticks([])
            plt.xlim([seuilmin,seuilmaxhisto])


        if len(therest)>2 :
            plt.set_cmap('jet')
#            plt.hist(npy.reshape(BB,ntot), bins=256, normed=True,range=[0.,2.],color='wheat')

#        plt.gray()            cax=plt.imshow(image, cmap=cm.coolwarm)

#    plt.colorbar(orientation="horizontal", fraction=0.05, aspect=40)
            vmax=seuilmax
            cbar = fig.colorbar(cax, ticks=[valmin, 0., valmax], orientation='horizontal', fraction=0.05, aspect=40)

#    print(malegende)

    if tagspe>0:
        ncol=npy.size(tabima,1)
        nlig=npy.size(tabima,0)
        print('plt.show dans visusar : image %d x %d'%(ncol,nlig))
        plt.show()

    return BB


def visusar(tabima,zparam=-999):
    """
    affichage d'images plutot radar.  Si image complexe : affichage de la valeur absolue

    plt.show() incorporé dans cette routine

    Arguments en entrée : 1 ou 2

        argument 1 : tableau 2D image

        argument 2 (facultatif) : facteur de la formule <<valeur moyenne + fak * écart type >>
        Si ce facteur est nul, l'image ne sera pas seuillée
        Si ce facteur est négatif, seuillage <<valeur moyenne - fak * écart type ; valeur moyenne + fak * écart type >>

        argument 3 (facultatif) :
            si nul : pas de plt.figure, ni de plt.show dans la procédure


    Argument en sortie : le tableau affiché (avec seuillage)

    Utilisez visusarZ (même syntaxe) pour éviter le plt.show()

    Utilisez visusarW (même syntaxe) pour n'avoir aucun affichage : on récupère le tableau qui aurait du être affiché
    """
# et si c'était un nom d'image ???

    coderetour = visusarbase(tabima,zparam,1)
    if type(tabima)== str :
        visusartitre(tabima)
    return coderetour

def visuinterfero(tabima,zparam=-999):
    """
    affichage d'images plutot radar.  Si image complexe : affichage de la valeur absolue

    plt.show() incorporé dans cette routine

    Arguments en entrée : 1 ou 2

        argument 1 : tableau 2D image

        argument 2 (facultatif) : facteur de la formule <<valeur moyenne + fak * écart type >>
        Si ce facteur est nul, l'image ne sera pas seuillée
        Si ce facteur est négatif, seuillage <<valeur moyenne - fak * écart type ; valeur moyenne + fak * écart type >>

        argument 3 (facultatif) :
            si nul : pas de plt.figure, ni de plt.show dans la procédure


    Argument en sortie : le tableau affiché (avec seuillage)

    Utilisez visusarZ (même syntaxe) pour éviter le plt.show()

    Utilisez visusarW (même syntaxe) pour n'avoir aucun affichage : on récupère le tableau qui aurait du être affiché
    """
# et si c'était un nom d'image ???

    coderetour = visusarbase(tabima,zparam,1,0,0,1)
    if type(tabima)== str :
        visusartitre(tabima)
    return coderetour

def visusarZ(tabima, *therest):
    nnn=-999
    itag=0
    if(len(therest)>0):
        nnn=therest[0]
    if(len(therest)>1):
        itag=therest[1]

    coderetour =    visusarbase(tabima,nnn,0,itag)

    if type(tabima)== str :
        visusartitre(tabima)

    return coderetour

        #
def visusarspectre(image,*therest):
    u'''
    Une image est passée en premier argument

    l'image et son spectre sont tracées

    le second argument (facultatif) est une légende
    '''

    global globalparamnotebook

    if  globalparamnotebook==0 :
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        visusarZ(image)
        plt.subplot(122)
        visusarZ(spyfftpack.fftshift(spyfftpack.fft2(image)))
        if(len(therest)>0):
            montitre=therest[0]
            plt.suptitle(montitre)
        plt.show()

    if  globalparamnotebook==1:
        if(len(therest)>0):
            montitre=therest[0]
        visusar(image)
        if(len(therest)>0):
            plt.suptitle(montitre)
        visusar(spyfftpack.fftshift(spyfftpack.fft2(image)))
        if(len(therest)>0):
            plt.suptitle(montitre)


def visuflicker(ima1, ima2) :
    """
    Deux images en paramètre
    cliquer sur la souris pour flicker l'image,
    entrer un caractère au clavier pour sortir


    ATTENTION : ne fonctionne pas avec un notebook
    """
    global globalparamnotebook
#
    if  globalparamnotebook==1 :
        print(u'NOTEBOOK : pas de flicker')
        visusar(ima1)
        visusar(ima2)

        return
# on canibalise visupile
    nkan0=2

    figpile=plt.figure()
    #tiilab.visusarZ(imatot[:,:,0])
    #plt.show()
    istop=0
    kut=0
    while istop==0 :
        if kut==0 :
            visusarZ(ima1)
        if kut==1 :
            visusarZ(ima2)
        plt.gcf().canvas.set_window_title(u'Image %d/%d \n Souris pour défiler, clavier pour arreter'%(kut+1, nkan0))
    #    plt.draw()
        figpile.canvas.draw()
        plt.show(block=False)
        test=plt.waitforbuttonpress()
        if test==False :
            print(u'Souris')
        if test==True :
                print(u'Clavier')
                istop=1
        kut=(kut+1)%nkan0
        if kut==nkan0 :
            istop=1

    plt.close(figpile)
#
######################################################
######################################################
        #
        # autres utilitaires
        #
######################################################
######################################################

def centrercentroidazi(fftbase):
    nlig=npy.size(fftbase,axis=0)
    sigspe=npy.sum(npy.abs(fftbase),axis=1)
    print(u'ICI %d'%npy.size(sigspe))
    tabz=npy.linspace(-math.pi,math.pi,nlig,endpoint=False)
    valspe=npy.dot(npy.cos(tabz),sigspe)+1j*npy.dot(npy.sin(tabz),sigspe)
    phase=npy.angle(valspe)
    ndecal=int(phase/math.pi*float(-nlig*0.5))
    fftcentroid=npy.roll(fftbase,ndecal,axis=0)
    print(u'ndecal = %d'%ndecal)
    return fftcentroid


def chirp_ers():
    fs=1.896e+7
    dt=1/fs;
    K=4.1889e+11;
    tt=3.712e-5

    ttz=tt/2.

    N=int(tt/dt)
    t=npy.linspace(-ttz,ttz,N)

    tq=npy.multiply(t,t)
    tq=-math.pi*1j*K*tq;
    sig=npy.exp(-tq)  # le signe - est pour la suite (image lausanne !!)
    sig=sig/math.sqrt(N)
    return sig

def synthese_range(imagename):
  mat = spio.loadmat(imagename)
  lambd = mat['p']['lambda'][0][0][0][0]
  h = mat['p']['h'][0][0][0][0]
  ts = mat['p']['ts'][0][0][0][0]
  chirp_rate = mat['p']['chirp_rate'][0][0][0][0]
  B = mat['p']['B'][0][0][0][0]
  AD = mat['p']['AD'][0][0][0][0]
  theta = mat['p']['theta'][0][0][0][0]
  L = mat['p']['L'][0][0][0][0]
  vplat = mat['p']['vplat'][0][0][0][0]
  PRF = mat['p']['PRF'][0][0][0][0]
  vec_range = mat['p']['vec_range'][0][0][0]
  vec_azimuth = mat['p']['vec_azimuth'][0][0][0]
  ref_range = mat['p']['ref_range'][0][0][:][:][:,0]
  data = mat['data']

  data_f = npy.fft.fft(data,n=len(vec_range),axis=0)
  S_f_rangeref = npy.fft.fft(ref_range,n=data.shape[0])
  compressed_data_f = npy.ones(data.shape,dtype=npy.complex)

  for T_idx in range(0,len(vec_azimuth)):
      compressed_data_f[:,T_idx] = data_f[:,T_idx]*npy.conj(S_f_rangeref)

  compressed_data = npy.fft.ifft(compressed_data_f,axis=0)
  return data, compressed_data

def synthese_azimuth(compressed_data, imagename):
  mat = spio.loadmat(imagename)
  lambd = mat['p']['lambda'][0][0][0][0]
  h = mat['p']['h'][0][0][0][0]
  ts = mat['p']['ts'][0][0][0][0]
  chirp_rate = mat['p']['chirp_rate'][0][0][0][0]
  B = mat['p']['B'][0][0][0][0]
  AD = mat['p']['AD'][0][0][0][0]
  theta = mat['p']['theta'][0][0][0][0]
  L = mat['p']['L'][0][0][0][0]
  vplat = mat['p']['vplat'][0][0][0][0]
  PRF = mat['p']['PRF'][0][0][0][0]
  vec_range = mat['p']['vec_range'][0][0][0]
  vec_azimuth = mat['p']['vec_azimuth'][0][0][0]
  ref_range = mat['p']['ref_range'][0][0][:][:][:,0]
  data = mat['data']
  vx = 0
  data_f = npy.fft.fft(compressed_data,axis=1,n=len(vec_azimuth))
  R0 = npy.mean(vec_range)
  Lsynt = lambd/L*R0
  T_ref = npy.arange(-Lsynt/(2*vplat),Lsynt/(2*vplat),1/PRF)
  vect_fd = npy.arange(0,len(vec_azimuth))/len(vec_azimuth)*PRF
  compressed_data_f = npy.ones(data.shape,dtype=npy.complex)
  compensated_data = npy.ones(data_f.shape,dtype=npy.complex)
  for t_idx in range(0,len(vec_range)):
      beta = 2*(vplat-vx)**2/(lambd*R0)
      chirp_az = npy.exp(-1j * npy.pi * beta * (T_ref ** 2))
      S_f_azimuthref = npy.fft.fft(chirp_az,n=data.shape[1])
      compressed_data_f[t_idx,:] = data_f[t_idx,:]*npy.conj(S_f_azimuthref)
      compensated_data[t_idx,:] = compressed_data_f[t_idx,:] * npy.exp(1j*2*npy.pi*Lsynt/(2*vplat)*vect_fd)
  compensated_data_t = npy.fft.ifftshift(npy.fft.ifft(compensated_data,axis=1),axes=0)
  return compensated_data_t
