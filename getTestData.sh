#!/bin/bash
# Autor: Michal Šedý <xsedym02@vutbr.cz>
# Last modification: 8.12.2021
# 1000 games per each combination
# players: kb.stei_at
#          dt.stei
#          kb.sdc_pre_at
#          kb.stei_dt
#          kb.stei_adt
#
# games: 4x kb.stei_at
#        4x dt.stei
#        4x kb.sdc_pre_at
#        4x kb.stei_dt
#        4x kb.stei_adt
#        2x kb.stei_at 2x dt.stei
#        2x kb.stei_at 2x kb.sdc_pre_at
#        2x kb.stei_at 2x kb.stei_dt
#        2x kb.stei_at 2x kb.stei_adt
#        2x dt.stei 2x kb.sdc_pre_at
#        2x dt.stei 2x kb.stei_dt
#        2x dt.stei 2x kb.stei_adt
#        2x kb.sdc_pre_at 2x kb.stei_dt
#        2x kb.sdc_pre_at 2x kb.stei_adt
#        2x kb.stei_dt 2x kb.stei_adt
#        1x kb.stei_at 1x kb.sdc_pre_at 1x kb.stei_dt 1x kb.stei_adt
#        1x dt.stei 1x kb.sdc_pre_at 1x kb.stei_dt 1x kb.stei_adt

function makeDirectories(){
    mkdir testData
    cd testData

    mkdir 4x
    cd 4x
    mkdir kb.stei_at
    mkdir dt.stei
    mkdir kb.sdc_pre_at
    mkdir kb.stei_dt
    mkdir kb.stei_adt
    cd ..

    mkdir 2x2x
    cd 2x2x
    mkdir kb.stei_at-dt.stei
    mkdir kb.stei_at-kb.sdc_pre_at
    mkdir kb.stei_at-kb.stei_dt
    mkdir kb.stei_at-kb.stei_adt
    mkdir dt.stei-kb.sdc_pre_at
    mkdir dt.stei-kb.stei_dt
    mkdir dt.stei-kb.stei_adt
    mkdir kb.sdc_pre_at-kb.stei_dt
    mkdir kb.sdc_pre_at-kb.stei_adt
    mkdir kb.stei_dt-kb.stei_adt
    cd ..

    mkdir 1x1x1x1x
    cd 1x1x1x1x
    mkdir kb.stei_at-kb.sdc_pre_at-kb.stei_dt-kb.stei_adt
    mkdir dt.stei-kb.sdc_pre_at-kb.stei_dt-kb.stei_adt
    cd ../..
}

function calculateData(){
    from=$1
    to=$2
    folder=$3
    AIs=$4
    for i in `seq $from $to`; do
        result=$(timeout -k 360s 360s python3 ./scripts/dicewars-ai-only.py --ai $AIs 2>&1 > ${folder}/${i}.bin)
        if [[ $?  -eq 0 ]]; then
            winner=$(echo $result | cut -d "'" -f2 | cut -d " " -f1)
            rate=$(echo $result | rev | cut -d "/" -f1 | rev)
            mv ${folder}/${i}.bin ${folder}/${i}_W-${winner}_R-${rate}.bin
            printf "%s\t%s\n"  "`date +"%T"`" "${i} - ${folder}"
        else
            printf "%s\tKILL\t%s\n"  "`date +"%T"`" "${i} - ${folder}"
        fi
    done
}

makeDirectories

#combinations of 4x
AI="kb.stei_at dt.stei kb.sdc_pre_at kb.stei_dt kb.stei_adt"
for ai in $AI; do
    from=$1
    step=$2
    to=$step
    calculateData $from $to "testData/4x/$ai" "${ai} ${ai}1 ${ai}2 ${ai}3"
    from=$(($from+$step))
    to=$(($to+$step))
done

# combinations of 2x2X
AI="kb.stei_at dt.stei kb.sdc_pre_at kb.stei_dt kb.stei_adt"
for i in `seq 1 4`; do
    outerAI=$(echo $AI | cut -d " " -f1)
    AI=$(echo $AI | cut -d " " -f2-)
    for ai in $AI; do
        from=$1
        step=$2
        to=$step
        calculateData $from $to "testData/2x2x/${outerAI}-${ai}" "${outerAI} ${outerAI}1 ${ai}1 ${ai}2"
        from=$(($from+$step))
        to=$(($to+$step))
    done
done


# combinations of 1x1x1x1x
AI="kb.stei_at kb.sdc_pre_at kb.stei_dt kb.stei_adt"
from=$1
step=$2
to=$step
calculateData $from $to "testData/1x1x1x1x/kb.stei_at-kb.sdc_pre_at-kb.stei_dt-kb.stei_adt" "${AI}"
from=$(($from+$step))
to=$(($to+$step))

# combinations of 1x1x1x1x
AI="dt.stei kb.sdc_pre_at kb.stei_dt kb.stei_adt"
from=$1
step=$2
to=$step
calculateData $from $to "testData/1x1x1x1x/dt.stei-kb.sdc_pre_at-kb.stei_dt-kb.stei_adt" "${AI}"
from=$(($from+$step))
to=$(($to+$step))

