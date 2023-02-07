#Program to automate results of nearby entities on basis of user input and location
#Import libraries
import numpy
import arcpy
import os

#Set Workspace
dir = r'\Proj2_SSCI\Proj2_SSCI.gdb'
arcpy.env.workspace = os.getcwd()+dir
arcpy.env.overwriteOutput = True

#Move Table to Geodatabase for further analysis
arcpy.conversion.TableToGeodatabase('G:\Coordinates_Proj2.csv', 'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb')

#Check
[val for val in arcpy.ListTables()]

#Project Data in Appropriate
arcpy.management.XYTableToPoint('Coordinates_Proj2', 'coordinates.shp',\
                                'Y', 'X', "")
arcpy.management.Project('G:\Python_Charm\Proj2_SSCI\coordinates.shp',
                         'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Coordinates_Rocklin',
                         arcpy.SpatialReference(3857))

#Delete Excess Rows + Fields in Dataset
fc = 'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Coordinates_Rocklin'
with arcpy.da.UpdateCursor(fc, 'x') as cursor:
    for row in cursor:
        if row[0] == 0:
            cursor.deleteRow()
    del cursor

arcpy.DeleteField_management('G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Coordinates_Rocklin',
                             ['Field5', 'Field6', 'Field7'])


#Create Feature Classes representing home, college, restaraunts, stores from original file
spatial_ref = arcpy.Describe("G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Coordinates_Rocklin").spatialReference
class Seg_Data():
    #User Determines the new feature class name
    def __init__(self):
        self.fc_name = input('Enter New Feature Class Name: ')
        self.spatial_ref =  arcpy.Describe("G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Coordinates_Rocklin").spatialReference

    #Creation of the empty feature class
    def create(self):
        arcpy.CreateFeatureclass_management('G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb',
                                            f'{self.fc_name}', "Point",
                                            'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Coordinates_Rocklin',
                                            "DISABLED", "DISABLED", spatial_ref)

    # User selects which values to Append to Empty Feature Class
    def append_values(self):
        fc = 'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Coordinates_Rocklin'
        fc1 = f'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\{self.fc_name}'

        #User given the unique values (home,college,stores, restataunts) applicable to append to new feature class
        with arcpy.da.SearchCursor(fc, 'type') as cursor: #From original DATASET
            unique_vals = sorted({row[0] for row in cursor})
        print(unique_vals)

        #Selected values appended to new feature class
        self.split_ = input('Enter value to append: ')
        with arcpy.da.SearchCursor(fc, ['x', 'y', 'type', 'name']) as cursor:
            for row in cursor:
                if row[2] == self.split_:
                    with arcpy.da.InsertCursor(fc1, ['x', 'y', 'type', 'name']) as cursor1:
                        cursor1.insertRow(row)

#Create Home Feature
s = Seg_Data()
s.create()
s.append_values()

#Create Groccery Store Feature
Gr_stores = Seg_Data()
Gr_stores.create()
Gr_stores.append_values()

#Create College Feature
College = Seg_Data()
College.create()
College.append_values()

#Create Restaurant Feature
Restaurants = Seg_Data()
Restaurants.create()
Restaurants.append_values()

for val in ['Home_Rocklin', 'Restaraunts_Rocklin', 'Groccery_Rocklin', 'College_Rocklin']:
    arcpy.management.XYTableToPoint(f'{val}', f'{val}.shp', \
                                    'Y', 'X', "")
    arcpy.management.Project(f'G:\Python_Charm\Proj2_SSCI\{val}.shp',
                             f'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\{val}_Projected',
                             arcpy.SpatialReference(3857))


#View restaraunts or groccery stores within x miles away from home or school
class Buffer:
    #User Determines starting location of user (Home/College)
    def __init__(self):
        while True:
            self.current_loc = input('Are you at Home or College?')
            if self.current_loc.lower() == 'home' or self.current_loc.lower() == 'college':
                break

    # User Determines whether to view (stores/restauraunts)
        while True:
            self.buff_select = input('View Stores or Restaurants? ')
            if self.buff_select.lower() == 'stores' or self.buff_select.lower() == 'restaurants':
                break

    #Create a 'X' Mile buffer (parameter passed in) of selected (stores/restaurants) from starting location
    def Create_Buffer(self, miles):
        self.miles = miles

        if self.buff_select.lower() == 'stores':
            self.in_fc = 'Groccery_Rocklin_Projected'
        elif self.buff_select.lower() == 'restaurants':
            self.in_fc = 'Restaraunts_Rocklin_Projected'

        arcpy.Buffer_analysis(self.in_fc, f'{self.in_fc}_Buffer{self.miles}mi', f'{self.miles} MILES')

    # Create an intersection analysis feature of the current location and the buffer created in the preceeding function
    def List_Near_Entities(self):
        return_ls = []
        if self.current_loc.lower() == 'home':
            self.infc = 'Home_Rocklin_Projected'
        elif self.current_loc.lower() == 'college':
            self.infc = 'College_Rocklin_Projected'

        arcpy.Intersect_analysis([f'{self.in_fc}_Buffer{self.miles}mi', f'{self.infc}'],
                                  f'{self.buff_select}_intersect_{self.current_loc}{self.miles}mi', 'ALL')


        #iterate through the intersection feature class and append names of nearby entities to return to user
        with arcpy.da.SearchCursor(f'{self.buff_select}_intersect_{self.current_loc}{self.miles}mi', 'name') as cursor:
            for row in cursor:
                return_ls.append(row[0])

        return return_ls


mile_buff = Buffer()
mile_buff.Create_Buffer(3)
mile_buff.List_Near_Entities()




#ARC NOTEBOOK
#Layer Symbology
dir = r'\Proj2_SSCI\Proj2_SSCI.aprx'
aprx = arcpy.mp.ArcGISProject(os.getcwd()+dir)
m = aprx.listMaps('Map')[0]
m.listLayers()
lyr = m.listLayers('Home_Rocklin')[0]
sym = lyr.symbology
if lyr.isFeatureLayer and hasattr(sym, "renderer"):
    sym.renderer.symbol.applySymbolFromGallery("Home", 2)
    lyr.symbology = sym
aprx.save()
del aprx



m = aprx.listMaps('Map')[0]
m.listLayers('Home_Rocklin')[0]

m = aprx.listMaps('Map')[0]
lyrs = m.listLayers()
for lyr in lyrs:
    sym = lyr.symbology
    if lyr.isFeatureLayer:
        if hasattr(sym, 'renderer'):
            print(lyr.name + ': ' + sym.renderer.type)







arcpy.ListTables()

arcpy.management.XYTableToPoint('Restaraunts_Rocklin1', 'Restaraunts_Rocklin1.shp',\
                                'Y', 'X', "")
arcpy.management.Project('G:\Python_Charm\Proj2_SSCI\Restaraunts_Rocklin1.shp',
                         'G:\Python_Charm\Proj2_SSCI\Proj2_SSCI.gdb\Restaraunts_Rocklin',
                         arcpy.SpatialReference(3857))
