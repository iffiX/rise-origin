<RSC Version="0.1">
    <Simulation>
        <IntegrationDtFrac>0.8</IntegrationDtFrac>
        <SoftbodyInternalDampingZ>1</SoftbodyInternalDampingZ>
        <SoftbodyContactDampingZ>0.8</SoftbodyContactDampingZ>
        <SoftbodyGlobalDampingZ>0.01</SoftbodyGlobalDampingZ>
        <RigidbodyConstraintIterations>10</RigidbodyConstraintIterations>
        <RigidbodyConstraintPositionCorrectionRatio>0.01</RigidbodyConstraintPositionCorrectionRatio>
        <RigidbodyLinearDampingZ>0.01</RigidbodyLinearDampingZ>
        <RigidbodyAngularDampingZ>0.01</RigidbodyAngularDampingZ>
        <RecordFrequency>20</RecordFrequency>
        <ControlFrequency>20</ControlFrequency>
        <StopCondition>
            t >= 10
        </StopCondition>
    </Simulation>
    <World>
        <GravAcc>-9.81</GravAcc>
        <Render>
            <Type>ConstantColor</Type>
            <Color>200, 200, 200</Color>
        </Render>
    </World>
    <Floor>
        <Data>
            <Type>Flat</Type>
        </Data>
        <Render>
            <Type>Opaque</Type>
            <Color>100, 150, 200</Color>
        </Render>
    </Floor>
    <!--    <Floor>-->
    <!--        <Data>-->
    <!--            <Type>Array</Type>-->
    <!--            <XSize>0.3</XSize>-->
    <!--            <YSize>0.3</YSize>-->
    <!--            <XValues>2</XValues>-->
    <!--            <YValues>2</YValues>-->
    <!--            <Height>-->
    <!--                0, 0.05, 0, 0.05-->
    <!--            </Height>-->
    <!--        </Data>-->
    <!--    </Floor>-->
    <Materials>
        <Material>
            <Name>material_0</Name>
            <Render>
                <Type>Opaque</Type>
                <Color>0, 0, 200</Color>
            </Render>
            <Mechanical>
                <ElasticModulus>1e5</ElasticModulus>
                <Density>1500</Density>
                <PoissonsRatio>0.35</PoissonsRatio>
                <UStatic>1</UStatic>
                <UDynamic>0.5</UDynamic>
                <MaxExpansion>0.5</MaxExpansion>
                <MinExpansion>-0.5</MinExpansion>
            </Mechanical>
        </Material>
        <Material>
            <Name>material_1</Name>
            <Render>
                <Type>Opaque</Type>
                <Color>0, 200, 0</Color>
            </Render>
            <Mechanical>
                <IsRigid>1</IsRigid>
                <ElasticModulus>1e5</ElasticModulus>
                <Density>1500</Density>
                <PoissonsRatio>0.35</PoissonsRatio>
                <UStatic>1</UStatic>
                <UDynamic>0.5</UDynamic>
                <MaxExpansion>0.5</MaxExpansion>
                <MinExpansion>-0.5</MinExpansion>
            </Mechanical>
        </Material>
        <Material>
            <Name>material_2</Name>
            <Render>
                <Type>Opaque</Type>
                <Color>200, 0, 0</Color>
            </Render>
            <Mechanical>
                <ElasticModulus>1e5</ElasticModulus>
                <Density>1500</Density>
                <PoissonsRatio>0.35</PoissonsRatio>
                <UStatic>1</UStatic>
                <UDynamic>0.5</UDynamic>
                <MaxExpansion>0.5</MaxExpansion>
                <MinExpansion>-0.5</MinExpansion>
            </Mechanical>
        </Material>
    </Materials>
</RSC>
