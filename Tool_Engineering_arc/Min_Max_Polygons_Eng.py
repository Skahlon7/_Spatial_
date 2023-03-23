# -*- coding: utf-8 -*-

import arcpy


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = "toolbox"

        # List of tool classes associated with this toolbox
        self.tools = [Max_polys, Min_polys]


class Max_polys(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Maximized Polygons"
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

        out_features = arcpy.Parameter(
        name = 'out_features',
        displayName = 'Output Features',
        datatype = "GPFeatureLayer",
        parameterType = "Required",
        direction = "Output"
        )

        num_features = arcpy.Parameter(
        name = "num_features",
        displayName = "Number of largest polygons to view",
        datatype = "GPLong",
        direction = "Input"
        )


        params = [input_features, out_features, num_features]
        return params

#     def isLicensed(self):
#         """Set whether tool is licensed to execute."""
#         return True
#
#     def updateParameters(self, parameters):
#         """Modify the values and properties of parameters before internal
#         validation is performed.  This method is called whenever a parameter
#         has been changed."""
#         return
#
#     def updateMessages(self, parameters):
#         """Modify the messages created by internal validation for each tool
#         parameter.  This method is called after internal validation."""
#         return
#
    def execute(self, parameters, messages):
        """The source code of the tool."""
        in_fc = parameters[0].valueAsText
        out_fc = parameters[1].valueAsText
        cnt = parameters[2].value

        id_ls, area_ls = [], []

        with arcpy.da.SearchCursor(in_fc, ['OID@', 'Shape_Area']) as cursor:
            for row in cursor:
                id_ls.append(row[0])
                area_ls.append(row[1])

        indexs = sorted(range(len(area_ls)), key= lambda i: area_ls[i], reverse=True)[:cnt]

        id_field = arcpy.da.Describe(in_fc)['OIDFieldName']
        sqlfield = arcpy.AddFieldDelimiters(in_fc, id_field)

        if cnt ==  1:
            sql_exp = f'{sqlfield} = {id_ls[0]}'
            arcpy.Select_analysis(in_fc, out_fc, sql_exp)
        else:
            sql_exp = f'{sqlfield} IN {tuple([id_ls[i] for i in indexs])}'
            arcpy.Select_analysis(in_fc, out_fc, sql_exp)

        arcpy.SetProgressorLabel("Processing data...")
        arcpy.AddMessage(f'Successfully located {cnt} maximized polygon(s)')

class Min_polys(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Minimized Polygons"
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

        out_features = arcpy.Parameter(
        name = 'out_features',
        displayName = 'Output Features',
        datatype = "GPFeatureLayer",
        parameterType = "Required",
        direction = "Output"
        )

        num_features = arcpy.Parameter(
        name = "num_features",
        displayName = "Number of smallest polygons to view",
        datatype = "GPLong",
        direction = "Input"
        )


        params = [input_features, out_features, num_features]
        return params

    def execute(self, parameters, messages):
        """The source code of the tool."""
        in_fc = parameters[0].valueAsText
        out_fc = parameters[1].valueAsText
        cnt = parameters[2].value

        id_ls, area_ls = [], []

        with arcpy.da.SearchCursor(in_fc, ['OID@', 'Shape_Area']) as cursor:
            for row in cursor:
                id_ls.append(row[0])
                area_ls.append(row[1])

        indexs = sorted(range(len(area_ls)), key= lambda i: area_ls[i])[:cnt]

        id_field = arcpy.da.Describe(in_fc)['OIDFieldName']
        sqlfield = arcpy.AddFieldDelimiters(in_fc, id_field)

        if cnt ==  1:
            sql_exp = f'{sqlfield} = {id_ls[0]}'
            arcpy.Select_analysis(in_fc, out_fc, sql_exp)
        else:
            sql_exp = f'{sqlfield} IN {tuple([id_ls[i] for i in indexs])}'
            arcpy.Select_analysis(in_fc, out_fc, sql_exp)

        arcpy.SetProgressorLabel("Processing data...")
        arcpy.AddMessage(f'Successfully located {cnt} minimized polygon(s)')


#original

#         sql_exp = f'{sqlfield} IN {tuple([id_ls[i] for i in indexs])}'
#         arcpy.Select_analysis(in_fc, out_fc, sql_exp