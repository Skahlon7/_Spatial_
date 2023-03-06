# -*- coding: utf-8 -*-

import arcpy

import random


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = "toolbox"

        # List of tool classes associated with this toolbox
        self.tools = [RandomSample]

class RandomSample(object):
    def __init__(self):
        self.label = "Random Sample"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        input_features = arcpy.Parameter(
        name = "input_features",
        displayName = "Input Features",
        datatype = "GPFeatureLayer",
        parameterType = "Required",
        direction = "Input"
        )

        output_features = arcpy.Parameter(
        name = "output_features",
        displayName = "Output Features",
        datatype = "GPFeatureLayer",
        parameterType = "Required",
        direction = "Output"
        )

        no_of_features = arcpy.Parameter(
        name = "number_of_features",
        displayName = "Number of Features",
        datatype = "GPLong",
        direction = "Input"
        )
        no_of_features.filter.type = "Range"
        no_of_features.filter.list = [1,  1000000000]

        parameters = [input_features, output_features, no_of_features]

        return parameters

    def execute(self, parameters, messages):
        inputfc = parameters[0].valueAsText
        outputfc = parameters[1].valueAsText
        outcount = parameters[2].value

        inlist = []
        with arcpy.da.SearchCursor(inputfc, "OID@") as cursor: #Iterate through rows to decipher ids
            for row in cursor:
                id = row[0]
                inlist.append(id)

        randomlist = random.sample(inlist, outcount) #random sample from id list

        desc = arcpy.da.Describe(inputfc)
        fldname = desc["OIDFieldName"] #locate OID column in file

        sqlfield = arcpy.AddFieldDelimiters(inputfc, fldname) #add sql expression to column holding id
        sqlexp = f'{sqlfield} IN {tuple(randomlist)}'

        arcpy.Select_analysis(inputfc, outputfc, sqlexp)


        