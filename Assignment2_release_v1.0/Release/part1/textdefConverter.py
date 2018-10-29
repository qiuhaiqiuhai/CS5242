import csv
import json

def toJSON(file_rows):
    res = '['
    for row in file_rows:
        res += '{"class":"com.worksap.company.hue.core.dto.TextDefDto","pl":"{\\\"key\\\":{\\\"class\\\":\\\"com.worksap.company.framework.textresource.TextResourceKey\\\",\\\"pl\\\":\\\"{\\\\\\\"textId\\\\\\\":\\\\\\\"%s\\\\\\\",\\\\\\\"locale\\\\\\\":\\\\\\\"%s\\\\\\\",\\\\\\\"plural\\\\\\\":\\\\\\\"\\\\\\\"}\\\"},\\\"value\\\":\\\"%s\\\",\\\"systemFlg\\\":null}"},\n'% (
        row[0], 'en', row[1])
        res += '{"class":"com.worksap.company.hue.core.dto.TextDefDto","pl":"{\\\"key\\\":{\\\"class\\\":\\\"com.worksap.company.framework.textresource.TextResourceKey\\\",\\\"pl\\\":\\\"{\\\\\\\"textId\\\\\\\":\\\\\\\"%s\\\\\\\",\\\\\\\"locale\\\\\\\":\\\\\\\"%s\\\\\\\",\\\\\\\"plural\\\\\\\":\\\\\\\"\\\\\\\"}\\\"},\\\"value\\\":\\\"%s\\\",\\\"systemFlg\\\":null}"},\n' % (
        row[0], 'ja', row[2])

    return res[:-2]+']';

def toCQL(file_rows):
    res = ''
    for row in file_rows:
        res+= "INSERT INTO core_text_def (tenant_id,text_id,locale,plural,system_flg,value) VALUES ('jill','%s','%s','',false,'%s');\n"%(row[0], 'en', row[1])
        res += "INSERT INTO core_text_def (tenant_id,text_id,locale,plural,system_flg,value) VALUES ('jill','%s','%s','',false,'%s');\n" % (
        row[0], 'ja', row[2])
    return res;


def load_csv(file_name):
    with open(file_name, encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        res = list(rows)
    return res


with open("Output.txt", "w", encoding='utf-8') as text_file:
    text_file.write(toJSON(load_csv('textdef.csv')))
    # text_file.write(toCQL(load_csv('textdef.csv')))