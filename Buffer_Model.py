#Buffer Model of schools <1 mile distance of Fault lines (Los Angeles County)
#Classification respecting earthquake risk
import arcpy
arcpy.env.workspace = 'G:\proj1_586\proj1_586.gdb'

#Question 1

fc_ls = ['private_schools', 'public_schools', 'EarthquakeFaults_USGS_CA_v2']
for val in fc_ls:
    arcpy.Clip_analysis(val, 'LA_boundary', f'{val}_LA')

#Question 2

arcpy.management.Merge(['public_schools_LA', 'private_schools_LA'], 'schools_LA')

#Check
arcpy.GetCount_management('schools_LA')
#<Result '3062'>
print(int(arcpy.GetCount_management('private_schools_LA')[0]) + int(arcpy.GetCount_management('public_schools_LA')[0]))
#<Result '3062'>

#Question 3
arcpy.Buffer_analysis('EarthquakeFaults_USGS_CA_v2_LA', 'EarthquakeFaults_LA_Buffer', '1 MILE')

arcpy.Clip_analysis('schools_LA', 'EarthquakeFaults_LA_Buffer', 'danger_schools')

#Question 4
#obtain information regarding which faultline is closest to respective schools
arcpy.analysis.Near('danger_schools', 'EarthquakeFaults_USGS_CA_v2_LA', method='Planar')

#Join field to reference closest fault line slip-rate with respect to LA Schools
arcpy.management.JoinField('danger_schools', 'NEAR_FID', 'EarthquakeFaults_USGS_CA_v2_LA', 'OBJECTID_1')

#Add Empty field to append class data
arcpy.management.AddField('danger_schools', 'Risk_Level', 'TEXT')

#Append class data to empty field on basis of slipcode
with arcpy.da.UpdateCursor('danger_schools', ['slipcode', 'Risk_Level']) as cursr:
    for row in cursr:
        if row[0] == 4:
            row[1] = 'Class D'
            cursr.updateRow(row)
        elif row[0] == 3:
            row[1] = 'Class C'
            cursr.updateRow(row)
        elif row[0] == 2:
            row[1] = 'Class B'
            cursr.updateRow(row)
        elif row[0] == 1:
            row[1] = 'Class A'
            cursr.updateRow(row)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
print('hello')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/EEE
